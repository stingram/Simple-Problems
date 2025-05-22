import torch

import triton
import triton.language as tl

@triton.jit
def _attn_fwd_inner(
            O_block,
            l_i,
            m_i,
            Q_block,
            K_block_ptr,
            V_block_ptr,
            block_index_q,
            softmax_scale,
            BLOCK_SIZE_Q: tl.constexpr,
            BLOCK_SIZE_KV: tl.constexpr,
            STAGE: tl.constexpr,
            offs_q: tl.constexpr,
            offs_kv: tl.constexpr,
            SEQ_LEN: tl.constexpr,
        ):
    # range of values handled by this stage
    if STAGE == 1:
        # from 0 to left of the diagonal
        lo, hi = 0, block_index_q * BLOCK_SIZE_Q
    elif STAGE == 2:
        # Used only for the block in which there is transition between non-masked and masked keys
        lo, hi = block_index_q * BLOCK_SIZE_Q, (block_index_q + 1) * BLOCK_SIZE_Q
        lo = tl.multiple_of(lo, BLOCK_SIZE_Q)
    else:
        # used only for non-causal attention
        lo, hi = 0, SEQ_LEN

    K_block_ptr = tl.advance(K_block_ptr, (0, lo))
    V_block_ptr = tl.advance(V_block_ptr, (lo, 0))
    
    # loop over k, v, and update accumulator
    for start_kv in range(lo, hi, BLOCK_SIZE_KV):
        # let the compiler know that start_n is a multiple of BLOCK_N, so the compiler can do optimizations
        # Telling the triton compiler this information helps improve its pipelining algorithm for the "for loop"
        start_kv = tl.multiple_of(start_kv, BLOCK_SIZE_KV)
        
        # compute qk
        K_block = tl.load(K_block_ptr)
        # we already have the pointers to read in K^t so we don't need to do that here
        QK_block = tl.dot(Q_block, K_block)
        
        if STAGE == 2:
            mask = offs_q[:, None] >= (start_kv + offs_kv[None, :])
            QK_block = QK_block * softmax_scale + tl.where(mask, 0, -1.0e6)
            m_ij = tl.maximum(m_i, tl.max(QK_block, 1))
            QK_block -= m_ij[:, None]
        else:
            # compute the maximum value of qk or keep the old max value
            m_ij = tl.maximum(m_i, tl.max(QK_block, 1) * softmax_scale)
            QK_block = QK_block * softmax_scale - m_ij[:, None]
            
        # Compute the exponential of each dot product, so now we are computing exp(qk_ij - m_ij)
        P_block = tl.math.exp(QK_block)
        
        # Compute the sum by rows of the attention scores
        l_ij = tl.sum(P_block, 1)
        
        # This is the correction factor for the previous l_i
        alpha = tl.math.exp(m_i - m_ij)
        
        # apply the correction factor to the previos l_i and add the new l_ij
        l_i = l_i * alpha + l_ij
        
        # now we can load V
        V_block = tl.load(V_block_ptr)
        
        # need to make sure P is the right data format
        P_block = P_block.to(tl.float16)
        
        # Now we can multiply P and V
        # This computes the following: O_new = P x V +O_old * alpha
        O_block = O_block * alpha[:, None]
        O_block = tl.dot(P_block, V_block, O_block) # O_block inside the dot here is the accumulator
        
        # now we update the correction factor for this kv block
        m_i = m_ij
        
        # now we move to the next block of K and V
        # We advance differently because the dimensions are ordered differently
        # and we need to increase SEQ_LEN by one KV BLOCK_SIZE
        # We've already transposed K, so we're working with K^t instead of K. That's 
        # why the dimensions are different between K and V here.
        V_block_ptr = tl.advance(V_block_ptr, (BLOCK_SIZE_KV, 0)) # V = [SEQ_LEN, HEAD_DIM]
        K_block_ptr = tl.advance(K_block_ptr, (0, BLOCK_SIZE_KV)) # K =[HEAD_DIM, SEQ_LEN]
        
    return O_block, l_i, m_i

@triton.autotune(
    [
        triton.Config(
            {"BLOCK_SIZE_Q": BLOCK_SIZE_Q, "BLOCK_SIZE_KV": BLOCK_SIZE_KV},
            num_stages=num_stages,
            num_warps=num_warps
        )
        for BLOCK_SIZE_Q in [64, 128] # this is just for experimenting, you can't know what is best given your hardware/software combination
        for BLOCK_SIZE_KV in [32, 64] # this is just for experimenting, you can't know what is best given your hardware/software combination
        for num_stages in ([3, 4, 7]) # software pipelining
        for num_warps in [2, 4] # A warp is a block of 32 threads that work cooperatively, running the same instruction at the same time
    ],
    key=["SEQ_LEN", "HEAD_DIM"] # all these combinations are tried every time either of these change and the optimal is chosen for each pair, optimal being the combination that takes least amount of time, or the combination the maximizes throughput
)
@triton.jit
def _attn_fwd(
    Q,  # BATCH_SIZE, NUM_HEADS, SEQ_LEN, HEAD_DIM,# Q[index_batch, index_head, :, :]
    K, # BATCH_SIZE, NUM_HEADS, SEQ_LEN, HEAD_DIM
    V, # BATCH_SIZE, NUM_HEADS, SEQ_LEN, HEAD_DIM
    softmax_scale,
    M, # BATCH_SIZE, NUM_HEADS, SEQ_LEN
    O, # BATCH_SIZE, NUM_HEADS, SEQ_LEN, HEAD_DIM
    stride_Q_batch,
    stride_Q_head,
    stride_Q_seq,
    stride_Q_dim,
    stride_K_batch,
    stride_K_head,
    stride_K_seq,
    stride_K_dim,
    stride_V_batch,
    stride_V_head,
    stride_V_seq,
    stride_V_dim,
    stride_O_batch,
    stride_O_head,
    stride_O_seq,
    stride_O_dim,
    BATCH_SIZE,
    NUM_HEADS: tl.constexpr,
    SEQ_LEN: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    BLOCK_SIZE_Q: tl.constexpr,
    BLOCK_SIZE_KV: tl.constexpr,
    STAGE: tl.constexpr,
):
    tl.static_assert(BLOCK_SIZE_KV <= HEAD_DIM)
    
    # This indicates which block in the sequence length to process
    block_index_q = tl.program_id(0)
    
    # This indicates which head and sequence to process. Each program is associated
    # with a single head of single sequence in a batch
    # This is our flattened 1D vector we need to convert into 2D indices
    index_batch_head = tl.program_id(1)
    # This indicates which sequence this program is associated with (each sequence has NUM_HEADS heads)
    index_batch = index_batch_head // NUM_HEADS
    # This indicates which head in the sequence
    index_head = index_batch_head % NUM_HEADS
    
    # This allows us to get the (SEQ_LEN, HEAD_DIM) sized block in the Q,K,V
    # by indexing it by batch and head
    qvk_offset = (
        index_batch.to(tl.int64)*stride_Q_batch
        + index_head.to(tl.int64)*stride_Q_head
    )
    
    Q_block_ptr = tl.make_block_ptr(
        base= Q + qvk_offset, # offset from first element marking the start of this block. Q[index_batch, index_head, block_index_q * BLOCK_SIZE_Q:, :]
        shape=(SEQ_LEN, HEAD_DIM), # full 2D shape of the slice
        strides=(stride_Q_seq, stride_Q_dim), # memory strides = how many elements to jump to move 1 step in each dimension
        offsets=(block_index_q * BLOCK_SIZE_Q, 0), # where to begin reading from within this 2D tile for this program, this is the outer for-loop in the algorithm
        block_shape=(BLOCK_SIZE_Q, HEAD_DIM), # shape of current block to load: rows x cols
        order=(1,0), # column-major layout: load HEAD_DIM fastest, SEQ_LEN slowest, not clear the impact on performance if order were to change
    )
    
    V_block_ptr = tl.make_block_ptr(
        base= V + qvk_offset, # offset from first element marking the start of this block.  V[index_batch, index_head, :, :]
        shape=(SEQ_LEN, HEAD_DIM), # full 2D shape of the slice
        strides=(stride_V_seq, stride_V_dim), # memory strides = how many elements to jump to move 1 step in each dimension
        offsets=(0, 0), # note 0 for the offset
        block_shape=(BLOCK_SIZE_KV, HEAD_DIM), # shape of current block to load: rows x cols
        order=(1,0), # column-major layout: load HEAD_DIM fastest, SEQ_LEN slowest, not clear the impact on performance if order were to change
    )

    K_block_ptr = tl.make_block_ptr(
        base= K + qvk_offset, # offset from first element marking the start of this block.  K[index_batch, index_head, :, :]
        shape=(HEAD_DIM, SEQ_LEN), # full 2D shape of the slice, note we swap dimensions here
        strides=(stride_K_dim, stride_K_seq), # memory strides = how many elements to jump to move 1 step in each dimension, note we swap dimensions here
        offsets=(0, 0), # note 0 for the offset
        block_shape=(HEAD_DIM, BLOCK_SIZE_KV), # shape of current block to load: rows x cols, note we swap dimensions here
        order=(1,0), # column-major layout: load HEAD_DIM fastest, SEQ_LEN slowest, not clear the impact on performance if order were to change
    )

    O_block_ptr = tl.make_block_ptr( # note this is same size,stride,shape, etc as Q
        base= O + qvk_offset, # offset from first element marking the start of this block. O[index_batch, index_head, block_index_q * BLOCK_SIZE_Q:, :]
        shape=(SEQ_LEN, HEAD_DIM), # full 2D shape of the slice
        strides=(stride_O_seq, stride_O_dim), # memory strides = how many elements to jump to move 1 step in each dimension
        offsets=(block_index_q * BLOCK_SIZE_Q, 0), # where to begin reading from within this 2D tile for this program, this is the outer for-loop in the algorithm
        block_shape=(BLOCK_SIZE_Q, HEAD_DIM), # shape of current block to load: rows x cols
        order=(1,0), # column-major layout: load HEAD_DIM fastest, SEQ_LEN slowest, not clear the impact on performance if order were to change
    )

    # offs_q: the offsets for the tokens in Q to process
    # we need a list of pointers to every single element for this program
    # note we skip up to this block
    offs_q = block_index_q * BLOCK_SIZE_Q + tl.arange(0, BLOCK_SIZE_Q)
    
    # offs_kv: the offsets for the tokens in the K and V sequence to process
    # note we don't skip anything
    offs_kv = tl.arange(0, BLOCK_SIZE_KV)
    
    # for each block of Q*K^t, we need to have maximum of each row and normalization factor for each row
    m_i = tl.zeros([BLOCK_SIZE_Q], dtype=tl.float32) - float("inf")
    
    # l_i: the running sum. We have one for each row (as we sum the attention scores by rows)
    # The +1 is to make the "log" stable since later we will use l_i to compute logsumexp
    l_i = tl.zeros([BLOCK_SIZE_Q], dtype=tl.float32) + 1.0
    
    # acc: the accumulator for the houtput, which is a group of rows of the matrix
    # for this program, 1 block of rows
    O_block = tl.zeros([BLOCK_SIZE_Q, HEAD_DIM], dtype=tl.float32)
    
    # load the blocks of Q: it will stay in SRAM throughout
    Q_block = tl.load(Q_block_ptr)  
    
    # now we need to call inner for-loop of flash attention algorithm

    # The reason we don't fuse the for loop to the left of the diagonal with the one exactly on the diagonal 
    # for the causal attention is to optimize the pipelining that Triton does.

    # Stage: 3 if causal, else 1
    if STAGE == 1 or STAGE == 3:
        # this step runs for non-causal attention or for the blocks to the left of the diagonal in the causal attention
        O_block, l_i, m_i = _attn_fwd_inner(
            O_block,
            l_i,
            m_i,
            Q_block,
            K_block_ptr,
            V_block_ptr,
            block_index_q,
            softmax_scale,
            BLOCK_SIZE_Q,
            BLOCK_SIZE_KV,
            4 - STAGE,
            offs_q,
            offs_kv,
            SEQ_LEN,
        )
    if STAGE == 3:
        # this step runs for the blocks to the right of the diagonal in the causal attention
        O_block, l_i, m_i = _attn_fwd_inner(
            O_block,
            l_i,
            m_i,
            Q_block,
            K_block_ptr,
            V_block_ptr,
            block_index_q,
            softmax_scale,
            BLOCK_SIZE_Q,
            BLOCK_SIZE_KV,
            2,
            offs_q,
            offs_kv,
            SEQ_LEN,
        )
    # epilogue
    m_i += tl.math.log(
        l_i
    )  # This is needed to compute the logsumexp for the backwards pass
    O_block = O_block / l_i[:, None]
    m_ptrs = M + index_batch_head * SEQ_LEN + offs_q
    tl.store(m_ptrs, m_i)
    tl.store(O_block_ptr, O_block.to(O.type.element_ty))

@triton.jit
def _attn_bwd_preprocess(
    O,
    dO,
    D, # (BATCH_SIZE, NUM_HEADS, SEQ_LEN)
    SEQ_LEN,
    BLOCK_SIZE_Q: tl.constexpr,
    HEAD_DIM: tl.constexpr,
):
    # We can use Q and O interchangeably here because O and Q have the
    # exact same shape. Actually, it is O that depends on the shape of Q.
    block_index_q = tl.program_id(0) # which group of vectors in O
    # we need to get list pointers for this program to process
    # 33,34,35,36 - this tell us which vectors in the output matrix among
    # all in the O-matrix this program is going to work with
    offs_q = block_index_q * BLOCK_SIZE_Q + tl.arange(0, BLOCK_SIZE_Q)
    # Tell us which batch and which head in each batach this particular
    # program is going to work with
    index_batch_head = tl.program_id(1)
    # need to load all dimensions of each vector
    # We just divide on sequence length dimension
    offs_dim = tl.arange(0, HEAD_DIM)
    
    # we will not use make block pointer like we did in forward pass
    # we will work directly with indexing by using strides
     
    # Load a single block of BLOCK_SIZE_Q rows of O
    O_block = tl.load( # O [BLOCK_SIZE, NUM_HEADS, SEQ_LEN, HEAD_DIM]
        O
        + index_batch_head * HEAD_DIM * SEQ_LEN
        + offs_q[:, None] * HEAD_DIM
        + offs_dim[None, :]
    ) # Size is (BLOCK_SIZE_Q, HEAD_DIM)

    # Load a single block of BLOCK_SIZE_Q rows of dO
    dO_block = tl.load( # dO [BLOCK_SIZE, NUM_HEADS, SEQ_LEN, HEAD_DIM]
        dO
        + index_batch_head * HEAD_DIM * SEQ_LEN
        + offs_q[:, None] * HEAD_DIM
        + offs_dim[None, :]
    ).to(tl.float32) # Size is (BLOCK_SIZE_Q, HEAD_DIM)
    
    # Compute the D block, sum is for each row
    D_block = tl.sum(dO_block * O_block, axis=1) # Shape: (BLOCK_SIZE_Q, )
    
    # Store the D block
    D_block_ptrs = D + index_batch_head * SEQ_LEN + offs_q
    tl.store(D_block_ptrs, D_block)
    
    # TODO - FINISH THIS

@triton.jit
def _attn_bwd_dk_dv(
    Q,
    K,
    V,
    softmax_scale,
    dO,
    dQ,
    dK,
    dV,
    M,
    D,
    stride_batch,
    stride_head,
    stride_seq,
    stride_dim,
    NUM_HEADS,
    SEQ_LEN,
    BLOCK_Q: tl.constexpr,
    BLOCK_KV: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    STAGE: tl.constexpr,
    num_warps,
    num_stages,
    ):
  index_batch_head = tl.program_id(2)
  index_batch = index_batch_head // NUM_HEADS
  index_head = index_batch_head % NUM_HEADS
  offset_batch_head = (stride_batch * index_batch + stride_head * index_head).to(
      tl.int64
  )
  # This is the offset that allows us to select the right sequence given
  # batch and head
  offset_batch_head_seq = (index_batch_head * SEQ_LEN).to(tl.int64)

  # Make sure the pointers are in the right place w.r.t. batch and head
  # The reason we don't access the blocks through make_block_ptr is because
  # we need to use the range of offsets to apply the masking
  Q += offset_batch_head
  K += offset_batch_head # [B, NUM_HEADS, SEQ_LEN, HEAD_DIM] -> [0, 0, SEQ_LEN, HEAD_DIM] -> [0,0,start_kv:start_kv+BLOCK_KV, 0:HEAD_DIM]
  V += offset_batch_head
  dO += offset_batch_head
  dQ += offset_batch_head
  dK += offset_batch_head
  dV += offset_batch_head
  
  # make sure pointers are in the right place w.r.t. batch, head, and sequence
  M += offset_batch_head_seq
  D += offset_batch_head_seq
  
  # load scales
  offs_dim = tl.arange(0, HEAD_DIM)
  
  index_block_kv = tl.program_id(0)
  start_kv = index_block_kv * BLOCK_KV
  
  offs_kv = start_kv + tl.arange(0, BLOCK_KV)
  
  dV_block = tl.zeros([BLOCK_KV, HEAD_DIM], dtype=tl.float32)
  dK_block = tl.zeros([BLOCK_KV, HEAD_DIM], dtype=tl.float32)
  
  # load K and V: they stay  in SRAM throughout the inner loop
  K_block = tl.load(
      K + offs_kv[:, None]* stride_seq + offs_dim[None, :] * stride_dim
  ) # Shape: (BLOCK_KV, HEAD_DIM)
  
  V_block = tl.load(
      V + offs_kv[:, None]* stride_seq + offs_dim[None, :] * stride_dim
  ) # Shape: (BLOCK_KV, HEAD_DIM)
  
  # Vector of pointers
  offs_q = tl.arange(0, BLOCK_Q) # 0,128,2*128,3*128,4,5,6,7, assuming head dim is 128 
  
  # Example assuming HEAD_DIM is 4, and for the sake of the example, assume this is for
  # accessing Q, not Q^t
  # so offs_q is a vector starting [0,BLOCK_Q) and then we multiply each element by HEAD_DIM
  # offs_q
  # ||
  # ||
  # ||
  # ||       offs_dim (which has HEAD_DIM number of columns)
  # ||        ||
  # ||        ||
  # \/        \/
  # 0*4 + (0,1,2,3) = (0,  1, 2, 3) # 1st Query Vector
  # 1*4 + (0,1,2,3) = (4,  5, 6, 7) # 2nd Query Vector
  # 2*4 + (0,1,2,3) = (8,  9,10,11) # 3rd Query Vector
  # 3*4 + (0,1,2,3) = (12,13,14,15) # 4th Query Vector
  
  # In memory these vectors are stored as
  # (0,  1, 2, 3) (4,  5, 6, 7)  (8,  9,10,11) (12,13,14,15)
  
  # 
  
  
  
  
  # We access Q as a transposed array, so that's why we treat offs_q as a column vector
  # and offs_dim as a row vector. This is equivalent to doing:
  # offs_q.unsqueeze(1) and the number of columns is detemined when added to offs_dim,
  # so it's a combination of unsqueeze and broadcasting, each column in a row will have
  # the same value
  # stride_seq tells us how many elements we need to skip to go from one query
  # vector to the next because each stride of the sequence dimension will be the size
  # of HEAD_DIM. To go from one query vector to the next, you need to go forward
  # by HEAD_DIM elements
  # q_ptrs = Q + offs_q[:,None]*stride_seq + offs_dim[:, None] * stride_dim
  # qT_ptrs = tl.trans(q_ptrs)
  # We point to the first BLOCK_Q_ rows of Q for both the qT and dO pointers,
  # inside the for loop loop we will move forward by BLOCK_Q rows at each
  # iteration
  # This lets us create a virtual tensor with the right shape that we want to "visualize"
  # When you work with a tensor layout in memory you can always view it as whatever shape you 
  # like. Reshaping is always free
  qT_ptrs = Q + offs_q[None, :] * stride_seq + offs_dim[:, None] * stride_dim
  dO_ptrs = dO + offs_q[:, None] * stride_seq + offs_dim[None, :] * stride_dim
  
  # Iterate over the sequence dimension of the query
  # in the query we need to go through all the seq_len dimension
  # because we select the key we want to work with
  # Here we are fixing key and go through all the queries
  # but for the query we need to start from zero and go until sequence length
  curr_q = 0
  num_steps = SEQ_LEN // BLOCK_Q
  for blk_idx in range(num_steps):
      # load a block of Q
      qT_block = tl.load(qT_ptrs)
      # load logsumexp values for the queries in the current block
      offs_q = curr_q + tl.arange(0, BLOCK_Q)
      m = tl.load(M + offs_q)
      
      # This gives us (QK^t)^t = (K^t)^t(Q^t) = K(Q^t) = S^t
      QK_T_block = softmax_scale * tl.dot(K_block, qT_block)
      
      # We apply the softmax by using logsumexp trick
      P_T_block = tl.math.exp(QK_T_block - m[None, :])
      
      if STAGE == 3:
          # Autoregressive masking.
          # mask is True for all values that we DO NOT NEED TO BE MASKED
          mask_block = (
              offs_q[None, :] >= offs_kv[:, None]
          ) # Shape: (BLOCK_KV1, BLOCK_Q1)
          # Replace all the masked values with 0.
          # In this case we do not need to mask with -Inf before applying the softmax since we already computed the normalization factors (stored in "m")
          P_T_block = tl.where(mask_block, P_T_block, 0.0)
          
      dO_block = tl.load(dO_ptrs)
      # According to the formula: dV_new = dV_old + P^t * dO, where * is matrix multiplication
      dV_block += tl.dot(P_T_block.to(tl.float16), dO_block)
      
      # Delta = rowsum(O * dO) where * is the element-wise product
      Di = tl.load(D + offs_q)
      
      # dP = dO * V^t, so dP^t = V * dO^t
      # Where * is matrix multiplication
      dpT_block = tl.dot(V_block, tl.trans(dO_block)).to(tl.float32)
      
      # We know that dS = P * (dP - Delta), so dS^t = P^t * (dP^t - Delta^t)
      dS_T_block = P_T_block * (dpT_block - Di[None, :])
      dS_T_block = dS_T_block.to(tl.float16)
      
      # According to the formula on the paper: dK_new = dK_old + dS^t x Q
      dK_block += softmax_scale * tl.dot(dS_T_block, tl.trans(qT_block))
      # Increment pointers
      curr_q += BLOCK_Q
      qT_ptrs += BLOCK_Q * stride_seq
      dO_ptrs += BLOCK_Q * stride_seq
      
  # Write back dV
  dv_block_ptrs = dV + offs_kv[:, None] * stride_seq + offs_dim[None, :] * stride_dim
  tl.store(dv_block_ptrs, dV_block)

  # Write back dK
  dk_block_ptrs = dK + offs_kv[:, None] * stride_seq + offs_dim[None, :] * stride_dim
  tl.store(dk_block_ptrs, dK_block)
  

@triton.jit
def _attn_bwd_dq(
    Q,
    K,
    V,
    softmax_scale,
    dO,
    dQ,
    dK,
    dV,
    M,
    D,
    stride_batch,
    stride_head,
    stride_seq,
    stride_dim,
    NUM_HEADS,
    SEQ_LEN,
    BLOCK_Q: tl.constexpr,
    BLOCK_KV: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    STAGE: tl.constexpr,
    num_warps,
    num_stages,
):
  ####### SAME INITIAL CODE AS OTHER FUNCTION for dV and dK ##############################
  index_batch_head = tl.program_id(2)
  index_batch = index_batch_head // NUM_HEADS
  index_head = index_batch_head % NUM_HEADS
  offset_batch_head = (stride_batch * index_batch + stride_head * index_head).to(
      tl.int64
  )
  # This is the offset that allows us to select the right sequence given
  # batch and head
  offset_batch_head_seq = (index_batch_head * SEQ_LEN).to(tl.int64)

  # Make sure the pointers are in the right place w.r.t. batch and head
  # The reason we don't access the blocks through make_block_ptr is because
  # we need to use the range of offsets to apply the masking
  Q += offset_batch_head
  K += offset_batch_head # [B, NUM_HEADS, SEQ_LEN, HEAD_DIM] -> [0, 0, SEQ_LEN, HEAD_DIM] -> [0,0,start_kv:start_kv+BLOCK_KV, 0:HEAD_DIM]
  V += offset_batch_head
  dO += offset_batch_head
  dQ += offset_batch_head
  dK += offset_batch_head
  dV += offset_batch_head
  
  # make sure pointers are in the right place w.r.t. batch, head, and sequence
  M += offset_batch_head_seq
  D += offset_batch_head_seq
  ####### SAME INITIAL CODE AS OTHER FUNCTION for dV and dK ##############################

  # load scales
  offs_dim = tl.arange(0, HEAD_DIM)
  
  index_block_kv = tl.program_id(0)
  
  start_q = index_block_kv * BLOCK_Q
  offs_q = start_q + tl.arange(0, BLOCK_Q) # example: 100, 101, 102, ..., 100 + BLOCK_Q - 1
  
  Q_block = tl.load(Q + offs_q[:, None] * stride_seq + offs_dim[None, :] * stride_dim)
  dQ_block = tl.zeros([BLOCK_Q, HEAD_DIM], dtype=tl.float32)
  dO_block = tl.load(
      dO + offs_q[:, None] * stride_seq + offs_dim[None, :] * stride_dim
  )
  
  M_block = tl.load(M + offs_q)
  M_block = M_block[:, None]
  
  offs_kv = tl.arange(0, BLOCK_KV)
  
  # We access the K and V as transposed blocks
  kT_ptrs = K + offs_kv[None, :] * stride_seq + offs_dim[:, None] * stride_dim
  vT_ptrs = V + offs_kv[None, :] * stride_seq + offs_dim[:, None] * stride_dim

  Di = tl.load(D + offs_q)
  
  curr_kv = 0
  num_steps = SEQ_LEN // BLOCK_KV
  for blk_idx in range(num_steps):
      K_T_block = tl.load(kT_ptrs)
      V_T_block = tl.load(vT_ptrs)
      
      QK_block = softmax_scale * tl.dot(Q_block, K_T_block) # recomputation of Q*K^t
      P_block = tl.math.exp(QK_block - M_block)
      
      if STAGE == 3:
        # autoregressive masking
        offs_kv = curr_kv + tl.arange(0,  BLOCK_KV)
        mask_block = offs_q[:, None] >= offs_kv[None, :]
        P_block = tl.where(mask_block, P_block, 0.0)
          
      # Compute dP and dS
      dP_block = tl.dot(dO_block, V_T_block).to(tl.float32)
      dS_block = P_block * (dP_block - Di[:, None])
      dS_block = dS_block.to(tl.float16)
      # Compute dQ
      dQ_block += softmax_scale * tl.dot(dS_block, tl.trans(K_T_block))
      
      # Increment Pointers
      curr_kv += BLOCK_KV
      kT_ptrs += BLOCK_KV * stride_seq
      vT_ptrs += BLOCK_KV * stride_seq
      
  dQ_block_ptrs = dQ + offs_q[:, None] * stride_seq + offs_dim[None, :] * stride_dim
  tl.store(dQ_block_ptrs, dQ_block)
  
  # 7:09:03

# Remember Q, K, V here are really W_q*Q, W_k*K, W_v*V. We don't care about optimizing
# thse initial matmuls for input Q,K,V
class TritonAttention(torch.autograd.Function):
    
    # note we don't want to recompute normalization factor or maximum value for each
    # row when executing the backward-pass, even though we're okay with recomputing
    # Q*K^t. We'll do a trick that allows us to save the two values as one. ctx is a
    # storage area for us to save information for the backward-pass
    @staticmethod
    def forward(ctx, Q, K, V, causal, softmax_scale):
        HEAD_DIM_Q, HEAD_DIM_K = Q.shape[-1], K.shape[-1]
        HEAD_DIM_V = V.shape[-1]
        
        BATCH_SIZE, NUM_HEADS, SEQ_LEN, HEAD_DIM = Q.shape
        assert HEAD_DIM_Q == HEAD_DIM_K and HEAD_DIM_K == HEAD_DIM_V
        
        # pre-allocate output tensor
        O = torch.empty_like(Q)
        stage = 3 if causal else 1

        # Define the Triton launch grid.
        # Grid is 3D: (blocks along sequence, batch*head dimension, depth=1)
        # - SEQ_LEN is split into blocks of size BLOCK_SIZE_Q (along the token/query dimension)
        # - Each (batch, head) pair operates independently
        # - Grid z-dimension is unused here
        grid = lambda args: (
            # ceil(SEQ_LEN / BLOCK_SIZE_Q) = How many blocks of Q we have
            triton.cdiv(SEQ_LEN, args["BLOCK_SIZE_Q"]),  # blocks over sequence length (token blocks)
            BATCH_SIZE * NUM_HEADS,                      # each (batch, head) pair gets its own program
            1                                            # 1 block in the z-dimension
        )
        
        # number of parallel programs: (BATCH * NUM_HEADS * NUM_BLOCKS_Q)
        
        # M is the logsumexp for the backward pass, one for each query, think of it as max for each row
        # This is the L_i on line 13 of the flash attention algorithm
        M = torch.empty(
            (BATCH_SIZE, NUM_HEADS, SEQ_LEN), device=Q.device, dtype=torch.float32
        )
        
        # Note: We don't pass BLOCK_SIZE_Q and BLOCK_SIZE_KV when calling the method because
        # they will be passed when we apply the Auto Tuning decorator
        _attn_fwd[grid](
            Q=Q,
            K=K,
            V=V,
            softmax_scale=softmax_scale,
            M=M,
            O=O,
            stride_Q_batch=Q.stride(0),
            stride_Q_head=Q.stride(1),
            stride_Q_seq=Q.stride(2),
            stride_Q_dim=Q.stride(3),
            stride_K_batch=K.stride(0),
            stride_K_head=K.stride(1),
            stride_K_seq=K.stride(2),
            stride_K_dim=K.stride(3),
            stride_V_batch=V.stride(0),
            stride_V_head=V.stride(1),
            stride_V_seq=V.stride(2),
            stride_V_dim=V.stride(3),
            stride_O_batch=O.stride(0),
            stride_O_head=O.stride(1),
            stride_O_seq=O.stride(2),
            stride_O_dim=O.stride(3),
            BATCH_SIZE=Q.shape[0],
            NUM_HEADS=Q.shape[1],
            SEQ_LEN=Q.shape[2],
            HEAD_DIM=HEAD_DIM_K,
            STAGE=stage,
        )
        # we want to compute gradients for Q,K,V
        # we also need to store M, O tensors
        ctx.save_for_backward(Q, K, V, O, M)
        ctx.grid = grid
        ctx.softmax_scale = softmax_scale
        ctx.HEAD_DIM = HEAD_DIM_K
        ctx.causal = causal
        
        # return O
        return O
        
    @staticmethod
    def backward(ctx, dO):
        Q, K, V, O, M = ctx.saved_tensors
        
        assert dO.is_contiguous()
        assert Q.stride() == K.stride() == V.stride() == O.stride() == dO.stride()
        dQ = torch.empty_like(Q)
        dK = torch.empty_like(K)
        dV = torch.empty_like(V)
        
        BATCH_SIZE, NUM_HEADS, SEQ_LEN = Q.shape[:3]
        NUM_WARPS, NUM_STAGES = 4, 3
        BLOCK_SIZE_MICRO, BLOCK_SIZE_MACRO = 32, 128
        
        preprocess_grid = (SEQ_LEN // BLOCK_SIZE_MACRO, BATCH_SIZE * NUM_HEADS)
        D = torch.empty_like(M) # Shape: (BATCH_SIZE, NUM_HEADS, SEQ_LEN)
        
        # Compute all elements Di
        _attn_bwd_preprocess[preprocess_grid](
            O=O,
            dO=dO,
            D=D,
            SEQ_LEN=SEQ_LEN,
            BLOCK_SIZE_Q=BLOCK_SIZE_MACRO,
            HEAD_DIM=ctx.HEAD_DIM,
        )
        # At this point pre-proessing is done and now we can do the two for loops

        # First, we fix KV and iterate through all Q 
        grid = (SEQ_LEN // BLOCK_SIZE_MACRO, 1, BATCH_SIZE * NUM_HEADS)
        stage = 3 if ctx.causal else 1

        _attn_bwd_dk_dv[grid](
            Q=Q,
            K=K,
            V=V,
            softmax_scale=ctx.softmax_scale,
            dO=dO,
            dQ=dQ,
            dK=dK,
            dV=dV,
            M=M,
            D=D,
            stride_batch=Q.stride(0),
            stride_head=Q.stride(1),
            stride_seq=Q.stride(2),
            stride_dim=Q.stride(3),
            NUM_HEADS=NUM_HEADS,
            SEQ_LEN=SEQ_LEN,
            BLOCK_Q=BLOCK_SIZE_MICRO,
            BLOCK_KV=BLOCK_SIZE_MACRO,
            HEAD_DIM=ctx.HEAD_DIM,
            STAGE=stage,
            num_warps=NUM_WARPS,
            num_stages=NUM_STAGES,
        )
        
        
        # Next, we fix Q, and iterate through all KV 
        _attn_bwd_dq[grid](
            Q=Q,
            K=K,
            V=V,
            softmax_scale=ctx.softmax_scale,
            dO=dO,
            dQ=dQ,
            dK=dK,
            dV=dV,
            M=M,
            D=D,
            stride_batch=Q.stride(0),
            stride_head=Q.stride(1),
            stride_seq=Q.stride(2),
            stride_dim=Q.stride(3),
            NUM_HEADS=NUM_HEADS,
            SEQ_LEN=SEQ_LEN,
            BLOCK_Q=BLOCK_SIZE_MACRO,
            BLOCK_KV=BLOCK_SIZE_MICRO,
            HEAD_DIM=ctx.HEAD_DIM,
            STAGE=stage,
            num_warps=NUM_WARPS,
            num_stages=NUM_STAGES,
        )
        
        return dQ, dK, dV, None, None      

    
        

def test_op(BATCH_SIZE, NUM_HEADS, SEQ_LEN, HEAD_DIM, causal, dtype=torch.float16):
    Q = (
        torch.empty(
            (BATCH_SIZE, NUM_HEADS, SEQ_LEN, HEAD_DIM), dtype=dtype, device="cuda:0"
        )
        .normal_(mean=0, std=0.5)
        .requires_grad_()
    )
    K = (
        torch.empty(
            (BATCH_SIZE, NUM_HEADS, SEQ_LEN, HEAD_DIM), dtype=dtype, device="cuda:0"
        )
        .normal_(mean=0, std=0.5)
        .requires_grad_()
    )
    V = (
        torch.empty(
            (BATCH_SIZE, NUM_HEADS, SEQ_LEN, HEAD_DIM), dtype=dtype, device="cuda:0"
        )
        .normal_(mean=0, std=0.5)
        .requires_grad_()
    )
    
    softmax_scale = 1/ (HEAD_DIM**0.5) # QK^t/sqrt(HEAD_DIM)
    dO = torch.randn_like(Q) # Needed for backward pass
    
    # reference implementation
    MASK = torch.tril(torch.ones((SEQ_LEN, SEQ_LEN), device="cuda:0"))
    P = torch.matmul(Q, K.transpose(2,3)) * softmax_scale
    if causal:
        P[:, :, MASK == 0] = float("-inf")
    P = torch.softmax(P.float(),dim=-1).half()
    ref_O = torch.matmul(P, V)
    ref_O.backward(dO)
    ref_dV, V.grad = V.grad.clone(), None
    ref_dK, K.grad = K.grad.clone(), None
    ref_dQ, Q.grad = Q.grad.clone(), None
    
    # Triton Implementation
    tri_out = TritonAttention.apply(Q, K, V, causal, softmax_scale).half()
    tri_out.backward(dO)
    tri_dV, V.grad = V.grad.clone(), None
    tri_dK, K.grad = K.grad.clone(), None
    tri_dQ, Q.grad = Q.grad.clone(), None
    
    # compare
    rtol = 0.0
    atol = 1e-2
    assert torch.allclose(ref_O, tri_out, atol=atol, rtol=rtol)
    assert torch.allclose(ref_dV, tri_dV, atol=atol, rtol=rtol)
    assert torch.allclose(ref_dK, tri_dK, atol=atol, rtol=rtol)
    assert torch.allclose(ref_dQ, tri_dQ, atol=atol, rtol=rtol)
    
if __name__ == "__main__":
    torch.empty(1, device="cuda:0")
    # test_op(BATCH_SIZE=8, NUM_HEADS=16, SEQ_LEN=4096, HEAD_DIM=64, causal=True)
    # test_op(BATCH_SIZE=8, NUM_HEADS=16, SEQ_LEN=4096, HEAD_DIM=64, causal=False)
    test_op(BATCH_SIZE=8, NUM_HEADS=16, SEQ_LEN=1024, HEAD_DIM=64, causal=True)
    test_op(BATCH_SIZE=8, NUM_HEADS=16, SEQ_LEN=1024, HEAD_DIM=64, causal=False)
    print("PASSED")