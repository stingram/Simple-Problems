


// class Solution:
    
//     def _build_dict(self, a: List[int]) -> Dict[int, bool]:
//         a_dict = {}
//         for num in a:
//             if num not in a_dict:
//                 a_dict[num] = True
//         return a_dict
    
//     def array_intersection(self, a1: List[int], a2: List[int]) -> List[int]:
        
//         # build dictionary of longest list
//         set_dict = {}
//         a_dict = self._build_dict(a1)
//         for n in a2:
//             if n in a_dict and n not in set_dict:
//                 set_dict[n] = True
        
//         return [k for k in set_dict.keys()]
    
// a1 = [4,9,5,9]
// a2 = [9,4,9,8,4]
// print(Solution().array_intersection(a1, a2))