import torch
import numpy as np
import tqdm as tqdm
from ddpm import DDPMSampler

WIDTH = 512
HEIGHT = 512
LATENT_WIDTH = WIDTH // 8
LATENT_HEIGHT = HEIGHT // 8


# TODO - derive this for practice
def rescale(x, old_range, new_range, clamp=False):
    old_min, old_max = old_range
    new_min, new_max = new_range

    x -= old_min
    x *= (new_max - new_min) / (old_max - old_min)
    x += new_min
    if clamp:
         x = x.clamp(new_min, new_max)
    return x

def get_time_embedding(timestep):
    # (160,)
    freqs = torch.pow(10000, -torch.arange(start=0,end=160, dtype=torch.float32) / 160)
    # (1,160)
    x = torch.tensor([timestep], dtype=torch.float32)[:, None] * freqs[None] # TODO - understand this
    # ( 1,320)
    return torch.cat([torch.cos(x), torch.sin(x)],dim=-1)


# for training maybe we'd have 1000 time-steps, but for inference we only need around 100

# unconditional_prompt - negative prompt, so if you put sofa in it, then there shouldn't be a sofa in the image
# cfg - classifier-free guidance
# top level function to generate an image 
# For classifier free guidance , the output is combined like the following:
# output = w*(out_conditioned-out_unconditioned)+out_unconditioned
# w - weight that indicates how much we want the model to pay attention to the conditioning signal (prompt)
# the weight in this function is "cfg_scale"
# so for each step we inference from the model twice, once with the prompt and once with the unconditional_prompt
# The unconditiona_prompt is usually an empty string
# strength - is how much we want the model to pay attention to the input image when generating the output image
# the higher the strength the more we  add noise leading to a more "Creative" model since it has more noise to remove

def generate(prompt: str,
             unconditional_prompt: str, # negative prompt or empty string
             input_image=None,
             strength=0.8,
             do_cfg=True,
             cfg_scale=7.5,
             sampler_name="ddpm",
             n_inference_steps=50,
             models={},
             seed=None,
             device=None,
             idle_device=None,
             tokenizer=None):
    
    with torch.no_grad():
        if not (0 < strength <= 1):
            raise ValueError("strength msut be between 0 adn 1.")
        if idle_device:
            to_idle: lambda x: x.to(idle_device)
        else:
            to_idle: lambda x: x
            
        generator = torch.Generator(device=device)
        if seed is None:
            generate.seed()
        else:
            generator.manual_seed(seed)
            
        clip = models["clip"]
        clip.to(device)
        
        if do_cfg:
            # convert prompt into tokens using the tokenizer
            cond_tokens = tokenizer.batch_encode_plus([prompt], padding="max_length", max_length=77).input_ids
            # (batch_size, seq_len)
            cond_tokens = torch.tensor(cond_tokens, dtype=torch.long, device=device)
            # now we run through CLIP
            # (batch_size, seq_len) -> (batch_size, seq_len, dim)
            cond_context = clip(cond_tokens)
            
            # do the same with an empty string for unconditional case
            uncond_tokens = tokenizer.batch_encode_plus([unconditional_prompt], padding="max_length", max_length=77).input_ids
            uncond_tokens = torch.tensor(uncond_tokens, dtype=torch.long, device=device)
            uncond_context = clip(uncond_tokens)
        
            # now we concat these embeddings
            # ( 2*batch_size, seq_len, dim) = (2*batch_size,77,768)
            context = torch.cat([cond_context, uncond_context])
        # we don't want to do classifier-free guidance, we only need to use prompt and we do one step through unet
        else:
            # convert it into a list of tokens
            tokens = tokenizer.batch_encode_plus([prompt], padding="max_length", max_length=77).input_ids
            tokens = torch.tensor(tokens, dtype=torch.long, device=device)
            # (batch_size, seq_len, dim) = (2*batch_size, 77, 768)
            context = clip(tokens)
        to_idle(clip) # offload this model to CPU since we are finished with it
        
        if sampler_name == "ddpm":
            sampler = DDPMSampler(generator)
            sampler.set_inference_steps(n_inference_steps)
        else:
            raise ValueError(f"Unknown sampler: {sampler_name}")
        
        # Latents that we will run through UNET
        latents_shape = (1,4, LATENT_HEIGHT, LATENT_WIDTH)
        
        # now what happens if the user specifies an input image?
        # so what if want image to image?
        # so lets say you already have a picture of a dog and your prompt is dog with glasses,
        # then ideally the model would output that same dog but now also with glasses
        if input_image:
            encoder = models["encoder"]
            encoder.to(device)
            
            input_image_tensor = input_image.resize((WIDTH, HEIGHT))
            input_image_tensor = np.array(input_image_tensor)
            # ( H, W, C)
            input_image_tensor = torch.tensor(input_image_tensor, dtype=torch.float32)
        
            # UNET wants every value in the input tensor to be between -1 and 1
            input_image_tensor = rescale(input_image_tensor, (0,255), (-1,1))
            input_image_tensor = input_image_tensor.unsqueeze(0)
            input_image_tensor = input_image_tensor.permute(0,3,1,2)
            
            # generate the N(0,1) sampled-noise for each pixel for the encoder
            encoder_noise = torch.randn(latents_shape, generator=generator, device=device)
            
            # now we can run the image through the encoder of the VAE to get latents
            latents = encoder(input_image_tensor, encoder_noise)
            
            # now we separately want to explicitly add noise to our latent
            sampler.set_strength(strength=strength)
            latents = sampler.add_noise(latents, sampler.timesteps[0])
            
            # no we don't need encoder anymore so we can move that to our "idle" device
            to_idle(encoder)
        
        # the user didn't give us an input image so we just start with random noise for 
        # our input image into the encoder, N(0,I)
        # (text to image)
        else:
            encoder_noise = torch.randn(latents_shape, generator=generator, device=device)
            
        diffusion = models["diffusion"]
        diffusion.to(device)
        
        
        # if maximum strength level is 1000, the minimum level would be 1
        # so it's 999 ... 0
        # or it's 1000 980 960 920 900 880 ... 0
        # depending on the number of steps
        # each number regardless, is a time-step that indicates the noise-level
        
        timesteps = tqdm(sampler.timesteps)
        for i, timestep in enumerate(timesteps):
            # (1 ,320)
            time_embedding = get_time_embedding(timestep).to(device)
            
            # (batch_size, 4, latent_H, latent_W)
            model_input = latents
            
            if do_cfg:
                # (batch_size, 4, latent_H, latent_W) -> # (2*batch_size, 4, latent_H, latent_W)
                # we double so we can have prompt and not have prompt
                model_input = model_input.repeat(2,1,1,1)
            
            # model_output is predicted noise by UNET
            model_output = diffusion(model_input, context, time_embedding)
            
            if do_cfg:
                output_cond, output_uncond = model_output.chunk(2)
                model_output = cfg_scale * (output_cond - output_uncond) + output_uncond
            
            # remove noise predicted by the eUNET
            latents = sampler.step(timestep, latents, model_output)
            
        to_idle(diffusion)
        
        decoder = models["decoder"]
        decoder.to(device)
        
        images = decoder(latents)
        to_idle(decoder)
        
        # at output of decoder we need to rescale back to "normal" range
        images = rescale(images, (-1,1), (0, 255), clamp=True)
        
        # (Batch_size, C, H, W) -> (Batch_size, H, W, C)
        images = images.permute(0,2,3,1)
        
        images = images.to("cpu",torch.int8).numpy()
        

        