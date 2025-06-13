import torch
import torch.nn.functional as F
import numpy as np

class PeakyLoss(object):
    """ PeakyLoss to avoid an uniform score map """

    def __init__(self, scores_th: float = 0.1):
        super().__init__()
        self.scores_th = scores_th

    def __call__(self, pred):
        b, c, h, w = pred['scores_map'].shape
        loss_mean = 0
        CNT = 0
        n_original = len(pred['score_dispersity'][0])
        for idx in range(b):
            scores_kpts = pred['scores'][idx][:n_original]
            valid = scores_kpts > self.scores_th
            loss_peaky = pred['score_dispersity'][idx][valid]

            loss_mean = loss_mean + loss_peaky.sum()
            CNT = CNT + len(loss_peaky)

        loss_mean = loss_mean / CNT if CNT != 0 else pred['scores_map'].new_tensor(0)
        assert not torch.isnan(loss_mean)
        return loss_mean
    

class ReliableLoss(object):
    """
    Reliable Loss as described in ALIKED paper.
    
    This loss constrains the score map by considering reliability based on 
    descriptor discriminativeness. Areas with low texture that are not 
    discriminative should have low scores and are considered unreliable.
    
    The reliability is computed based on the matching similarity vector,
    encouraging keypoints in areas with distinctive descriptors.
    """
    
    def __init__(self, t_rel: float = 1):
        """
        Args:
            t_rel: Temperature parameter for reliability softmax
            alpha: Weight for the reliability constraint term
            beta: Weight for the unreliability penalty term
        """
        super().__init__()
        self.t_rel = t_rel
        # self.alpha = alpha
        # self.beta = beta
    
    def compute_reliability(self, descriptor_A, descriptors_B):
        """
        Compute reliability r(pA, IB) based on matching similarity vector.
        
        Args:
            descriptor_A: (D,) descriptor for a single keypoint in image A
            descriptors_B: (NB, D) descriptors for all keypoints in image B
            
        Returns:
            reliability: Scalar reliability value (max of softmax probabilities)
        """
        # Compute similarity: sim(dA, DB) = DB * dA
        similarities = torch.matmul(descriptors_B, descriptor_A)  # (NB,)
        
        # Apply temperature and softmax: r = softmax(sim / t_rel)
        scaled_similarities = similarities / self.t_rel
        reliability_probs = F.softmax(scaled_similarities, dim=0)  # (NB,)
        
        # Take maximum probability as reliability measure
        # High max probability means the descriptor is distinctive
        reliability = torch.max(reliability_probs)
        
        return reliability
    
    def compute_reliability_bidirectional(self, descriptors_A, descriptors_B):
        """
        Compute reliability for all keypoints in both directions A->B and B->A.
        
        Args:
            descriptors_A: (NA, D) descriptors for keypoints in image A
            descriptors_B: (NB, D) descriptors for keypoints in image B
            
        Returns:
            reliability_A: (NA,) reliability values for keypoints in A w.r.t. B
            reliability_B: (NB,) reliability values for keypoints in B w.r.t. A
        """
        NA, NB = descriptors_A.shape[0], descriptors_B.shape[0]
        device = descriptors_A.device
        
        # Compute reliability A -> B
        reliability_A = torch.zeros(NA, device=device)
        for i in range(NA):
            descriptor_A = descriptors_A[i]  # (D,)
            reliability_A[i] = self.compute_reliability(descriptor_A, descriptors_B)
        
        # Compute reliability B -> A  
        reliability_B = torch.zeros(NB, device=device)
        for j in range(NB):
            descriptor_B = descriptors_B[j]  # (D,)
            reliability_B[j] = self.compute_reliability(descriptor_B, descriptors_A)
            
        return reliability_A, reliability_B
    
    def extract_keypoint_scores(self, score_map, keypoints):
        """
        Extract scores from score map at keypoint locations.
        
        Args:
            score_map: (H, W) score map
            keypoints: (N, 2) keypoint coordinates in normalized format [-1, 1]
            
        Returns:
            scores: (N,) scores at keypoint locations
        """
        H, W = score_map.shape
        N = keypoints.shape[0]
        device = keypoints.device
        
        scores = torch.zeros(N, device=device)
        
        for i, kpt in enumerate(keypoints):
            # Convert normalized coordinates to pixel coordinates
            x_norm, y_norm = kpt[0], kpt[1]
            x = int((x_norm + 1) * (W - 1) / 2)
            y = int((y_norm + 1) * (H - 1) / 2)
            
            # Check bounds
            if 0 <= x < W and 0 <= y < H:
                scores[i] = score_map[y, x]
            else:
                scores[i] = 0.0  # Out of bounds keypoints get 0 score
                
        return scores
    
    def compute_reliable_loss_single_pair(self, descriptors_A, descriptors_B, 
                                      keypoints_A, keypoints_B,
                                      score_map_A, score_map_B):
        if descriptors_A.shape[0] == 0 or descriptors_B.shape[0] == 0:
            return torch.tensor(0.0, device=descriptors_A.device, requires_grad=True)
        
        # Compute reliability
        reliability_A, reliability_B = self.compute_reliability_bidirectional(descriptors_A, descriptors_B)

        # Extract scores
        scores_A = self.extract_keypoint_scores(score_map_A, keypoints_A)  # (NA,)
        scores_B = self.extract_keypoint_scores(score_map_B, keypoints_B)  # (NB,)

        # Normalizing factors
        sum_scores_A = scores_A.sum()
        sum_scores_B = scores_B.sum()

        # Avoid division by zero
        loss_A = ((1.0 - reliability_A) * scores_A).sum() / (sum_scores_A + 1e-8)
        loss_B = ((1.0 - reliability_B) * scores_B).sum() / (sum_scores_B + 1e-8)

        # Average both directions
        total_loss = (loss_A + loss_B) / 2.0
        return total_loss
    
    def __call__(self, pred_A, pred_B):
        """
        Compute overall reliable loss for batch.
        
        Args:
            pred: Dictionary containing:
                - 'descriptors': List of (N, D) descriptor matrices for each image in batch
                - 'keypoints': List of (N, 2) keypoint coordinates for each image in batch  
                - 'scores_map': (B, C, H, W) score maps for all images
                    
        Returns:
            loss: Average reliable loss across batch
        """
        batch_size = len(pred_A['descriptors'])
        # print(f"Computing reliable loss for batch size: {batch_size}")
        
        total_loss = 0.0
        valid_pairs = 0
        
        # Process image pairs
        for batch_idx in range(0, batch_size):
        
                
            # Get data for image pair
            desc_A = pred_A['descriptors'][batch_idx]      # (NA, D)
            desc_B = pred_B['descriptors'][batch_idx]  # (NB, D)
            kpts_A = pred_A['keypoints'][batch_idx]        # (NA, 2)
            kpts_B = pred_B['keypoints'][batch_idx]    # (NB, 2)
            
            # Extract score maps (assuming single channel)
            score_map_A = pred_A['scores_map'][batch_idx, 0]       # (H, W)
            score_map_B = pred_B['scores_map'][batch_idx, 0]
            
            # Compute reliable loss for this pair
            pair_loss = self.compute_reliable_loss_single_pair(
                desc_A, desc_B, kpts_A, kpts_B, score_map_A, score_map_B
            )
            
            total_loss += pair_loss
            valid_pairs += 1
        
        if valid_pairs > 0:
            return total_loss / valid_pairs
        else:
            return torch.tensor(0.0, device=pred_A['descriptors'][0].device, requires_grad=True)



class SparseNRELoss(object):
    """
    Sparse Neural Reprojection Error Loss as described in ALIKED paper.
    
    This loss encourages matching keypoints to have similar descriptors while
    pushing non-matching keypoints to have distinct descriptors using cross-entropy
    between reprojection probability and matching probability vectors.
    """
    
    def __init__(self, t_des: float = 0.1):
        """
        Args:
            t_des: Temperature parameter controlling sharpness of matching probability
            valid_radius: Radius in pixels to consider a match as valid (for reprojection)
        """
        super().__init__()
        self.t_des = t_des
        
    # def compute_reprojection_probability(self, descriptors_A, descriptors_B, matches_AB):
    #     """
    #     Compute binary reprojection probability vector qr(pA, PB).
        
    #     Args:
    #         descriptors_A: (NA, 2) keypoints in image A (normalized coordinates)
    #         descriptors_B: (NB, 2) keypoints in image B (normalized coordinates)  
    #         matches_AB: (M, 2) matches between A and B, where matches_AB[i] = [idx_A, idx_B]
            
    #     Returns:
    #         reprojection_probs: List of (NB,) binary vectors for each keypoint in A
    #     """
    #     NA = descriptors_A.shape[0]
    #     NB = descriptors_B.shape[0]
    #     device = descriptors_A.device
        
    #     reprojection_probs = []
        
    #     for i in range(NA):
    #         # Create binary vector of size NB
    #         qr = torch.zeros(NB, device=device)
            
    #         # Find matches for keypoint i in image A
    #         match_mask = matches_AB[:, 0] == i
    #         if match_mask.any():
    #             matched_indices_B = matches_AB[match_mask, 1]
    #             qr[matched_indices_B] = 1.0
                
    #         reprojection_probs.append(qr)
        
    #     reprojection_probs_cpu = [prob.cpu() for prob in reprojection_probs]

            
    #     return torch.from_numpy(np.array(reprojection_probs_cpu))
    
    def compute_reprojection_probability(self, descriptors_A, descriptors_B, matches_AB):
        """
        Compute binary reprojection probability matrix qr(pA, PB).
        
        Args:
            descriptors_A: (NA, 2) keypoints in image A (normalized coordinates)
            descriptors_B: (NB, 2) keypoints in image B (normalized coordinates)  
            matches_AB: (M, 2) matches between A and B, where matches_AB[i] = [idx_A, idx_B]
            
        Returns:
            reprojection_probs: (NA, NB) binary matrix where reprojection_probs[i, j] = 1 
                                if keypoint i in A matches keypoint j in B, else 0
        """
        NA = descriptors_A.shape[0]
        NB = descriptors_B.shape[0]
        device = descriptors_A.device

        # Initialize zero matrix
        reprojection_probs = torch.zeros((NA, NB), device=device)

        # Unpack matched indices
        idx_A, idx_B = matches_AB[:, 0], matches_AB[:, 1]

        # Set 1 where matches exist
        reprojection_probs[idx_A, idx_B] = 1.0

        return reprojection_probs


    def compute_matching_probability(self, descriptor_A, descriptors_B):
        """
        Compute matching probability vector qm(dA, DB) using softmax of similarities.
        
        Args:
            descriptor_A: (D,) descriptor for a single keypoint in image A
            descriptors_B: (NB, D) descriptors for all keypoints in image B
            
        Returns:
            matching_prob: (NB,) matching probability vector
        """
        # Compute similarity: sim(dA, DB) = DB * dA
        similarities = torch.matmul(descriptors_B, descriptor_A)  # (NB,)
        
        # Apply temperature and softmax: qm = softmax((sim - 1) / t_des)
        scaled_similarities = (similarities - 1.0) / self.t_des
        matching_prob = F.softmax(scaled_similarities, dim=0)
        
        return matching_prob
    
    def compute_sparse_nre_loss_single_direction(self, descriptors_A, descriptors_B, 
                                               matches_AB):
        """
        Compute sparse NRE loss for keypoints A -> B direction.
        
        Args:
            descriptors_A: (NA, D) descriptors for keypoints in image A
            descriptors_B: (NB, D) descriptors for keypoints in image B
            keypoints_A: (NA, 2) keypoints in image A
            keypoints_B: (NB, 2) keypoints in image B
            matches_AB: (M, 2) matches between A and B
            
        Returns:
            loss: Average cross-entropy loss for all keypoints in A
        """
        if matches_AB.shape[0] == 0:
            return torch.tensor(0.0, device=descriptors_A.device, requires_grad=True)
            
        NA = descriptors_A.shape[0]
        # print(f"Computing sparse NRE loss for {NA} keypoints in A and {descriptors_B.shape[0]} keypoints in B")
        # print(f"Number of matches: {matches_AB.shape[0]}")
        # Get reprojection probabilities for all keypoints in A
        reprojection_probs = self.compute_reprojection_probability(
            descriptors_A, descriptors_B, matches_AB
        )
        
        total_loss = 0.0
        valid_count = 0
        
        for i in range(NA):
            # Check if keypoint i has any matches
            qr = reprojection_probs[i]  # (NB,)
            if qr.sum() == 0:  # No matches for this keypoint
                continue
                
            # Get descriptor for keypoint i in image A
            descriptor_A = descriptors_A[i]  # (D,)
            
            # Compute matching probability
            qm = self.compute_matching_probability(descriptor_A, descriptors_B)  # (NB,)
            
            # Compute cross-entropy loss: CE(qr || qm) = -sum(qr * log(qm))
            # Since qr is binary, this simplifies to -ln(qm[matched_indices])
            matched_indices = torch.nonzero(qr, as_tuple=True)[0]
            
            if len(matched_indices) > 0:
                # Sum over all matched keypoints (in case of multiple matches)
                ce_loss = -torch.log(qm[matched_indices] + 1e-8).mean()
                total_loss += ce_loss
                valid_count += 1
        
        if valid_count > 0:
            return total_loss / valid_count
        else:
            return torch.tensor(0.0, device=descriptors_A.device, requires_grad=True)

    # def compute_sparse_nre_loss_single_direction(self, descriptors_A, descriptors_B, matches_AB):
    #     """
    #     Compute sparse NRE loss for keypoints A -> B direction.
        
    #     Args:
    #         descriptors_A: (NA, D) descriptors for keypoints in image A
    #         descriptors_B: (NB, D) descriptors for keypoints in image B
    #         matches_AB: (M, 2) matches between A and B
            
    #     Returns:
    #         loss: Average cross-entropy loss for all keypoints in A
    #     """
    #     device = descriptors_A.device
    #     if matches_AB.shape[0] == 0:
    #         return torch.tensor(0.0, device=device, requires_grad=True)

    #     NA = descriptors_A.shape[0]
    #     NB = descriptors_B.shape[0]

    #     # Step 1: Get reprojection_probs matrix (NA, NB) binary
    #     reprojection_probs = self.compute_reprojection_probability(descriptors_A, descriptors_B, matches_AB)
    #     # Shape: (NA, NB)

    #     # Step 2: Compute matching probabilities qm for all descriptors_A at once
    #     # Assuming compute_matching_probability can be vectorized to handle batch:
    #     # Input: (NA, D), descriptors_B: (NB, D)
    #     # Output: (NA, NB) matching probabilities
        
    #     # If your compute_matching_probability works only for one descriptor at a time,
    #     # you should vectorize it for efficiency.
    #     qm = self.compute_matching_probability_batch(descriptors_A, descriptors_B)
    #     # Shape: (NA, NB)

    #     # Step 3: For numerical stability, add epsilon
    #     eps = 1e-8
    #     qm = torch.clamp(qm, min=eps, max=1.0)

    #     # Step 4: Compute cross entropy loss only for matched keypoints
    #     # For each keypoint i, gather qm[i, j] where reprojection_probs[i, j] == 1
    #     # Since reprojection_probs is binary mask, multiply element-wise and use log
        
    #     # Mask qm with reprojection_probs to zero out non-matches
    #     matched_qm = qm * reprojection_probs  # (NA, NB)

    #     # To avoid zeros when averaging, compute number of matches per keypoint
    #     matches_per_keypoint = reprojection_probs.sum(dim=1)  # (NA,)

    #     # Compute loss per keypoint: mean negative log likelihood over matches
    #     # sum over NB dimension, then divide by number of matches per keypoint
    #     # For keypoints with zero matches, avoid division by zero by masking later
        
    #     # sum(-log(qm)) per keypoint over matches
    #     loss_per_keypoint = -(matched_qm + eps).log() * reprojection_probs  # zeros outside matches
        
    #     # sum losses per keypoint
    #     sum_loss_per_keypoint = loss_per_keypoint.sum(dim=1)  # (NA,)

    #     # mean loss per keypoint (avoid division by zero)
    #     mean_loss_per_keypoint = sum_loss_per_keypoint / matches_per_keypoint.clamp(min=1.0)

    #     # Only consider keypoints with matches
    #     valid_mask = matches_per_keypoint > 0

    #     if valid_mask.sum() == 0:
    #         return torch.tensor(0.0, device=device, requires_grad=True)

    #     # Average loss over valid keypoints
    #     total_loss = mean_loss_per_keypoint[valid_mask].mean()

    #     return total_loss
    
    def __call__(self, pred_a, pred_b, correspondences=None):
        """
        Compute overall sparse NRE loss for both directions A->B and B->A.
        
        Args:
            pred: Dictionary containing:
                - 'descriptors': List of (N, D) descriptor matrices for each image in batch
                - 'keypoints': List of (N, 2) keypoint coordinates for each image in batch
                - 'matches': List of (M, 2) match indices for each pair in batch
                    or single (M, 3) tensor where first column is batch index
                    
        Returns:
            loss: Average sparse NRE loss across batch and both directions
        """
        if isinstance(pred_a['descriptors'], list):
            return self._compute_loss_from_list(pred_a, pred_b, correspondences)
        else:
            raise NotImplementedError(
                "SparseNRELoss currently only supports input as a list of dictionaries per batch. ")
            # return self._compute_loss_from_tensor(pred)
    
    def _compute_loss_from_list(self, pred_A, pred_B, correspondences):
        """Handle case where matches are provided as list per batch."""
        batch_size = len(pred_A['descriptors'])
        total_loss = 0.0
        valid_pairs = 0
        
        for batch_idx in range(0, batch_size):
                
            # Get data for image pair
            desc_A = pred_A['descriptors'][batch_idx]      # (NA, D)
            desc_B = pred_B['descriptors'][batch_idx]  # (NB, D)
            # kpts_A = pred_A['keypoints'][batch_idx]        # (NA, 2)
            # kpts_B = pred_A['keypoints'][batch_idx]    # (NB, 2)
           
            matches = correspondences[batch_idx]['matches']  # (M, 2) matches from A to B
            if matches.shape[0] == 0:
                continue
            
            # Compute loss A -> B
            loss_AB = self.compute_sparse_nre_loss_single_direction(
                desc_A, desc_B, matches
            )

            # print("LOSS A -> B:", loss_AB.item())
            
            # Compute loss B -> A (flip match indices)
            matches_BA = torch.stack([matches[:, 1], matches[:, 0]], dim=1)
            loss_BA = self.compute_sparse_nre_loss_single_direction(
                desc_B, desc_A, 
                matches_BA
            )
            
            # Average both directions
            pair_loss = (loss_AB + loss_BA) / 2.0
            total_loss += pair_loss
            valid_pairs += 1
        
        if valid_pairs > 0:
            return total_loss / valid_pairs
        else:
            return torch.tensor(0.0, device=pred_A['descriptors'][0].device, requires_grad=True)
    
    def _compute_loss_from_tensor(self, pred):
        """Handle case where matches are provided as single tensor with batch indices."""
        matches_with_batch = pred['matches']  # (M, 3) where first column is batch_idx
        
        if matches_with_batch.shape[0] == 0:
            return torch.tensor(0.0, device=pred['descriptors'][0].device, requires_grad=True)
        
        total_loss = 0.0
        valid_pairs = 0
        
        # Group matches by batch pairs
        unique_batches = torch.unique(matches_with_batch[:, 0])
        
        for batch_pair_idx in unique_batches:
            batch_idx = int(batch_pair_idx.item())
            
            # Assume pairs are (0,1), (2,3), etc.
            img_A_idx = batch_idx * 2
            img_B_idx = batch_idx * 2 + 1
            
            if img_B_idx >= len(pred['descriptors']):
                continue
            
            # Filter matches for this batch pair
            batch_mask = matches_with_batch[:, 0] == batch_pair_idx
            matches = matches_with_batch[batch_mask, 1:3]  # (M, 2)
            
            # Get data for image pair
            desc_A = pred['descriptors'][img_A_idx]
            desc_B = pred['descriptors'][img_B_idx]
            # kpts_A = pred['keypoints'][img_A_idx]
            # kpts_B = pred['keypoints'][img_B_idx]
            
            # Compute bidirectional loss
            loss_AB = self.compute_sparse_nre_loss_single_direction(
                desc_A, desc_B, matches
            )
            
            matches_BA = torch.stack([matches[:, 1], matches[:, 0]], dim=1)
            loss_BA = self.compute_sparse_nre_loss_single_direction(
                desc_B, 
                desc_A, 
                matches_BA
            )
            
            pair_loss = (loss_AB + loss_BA) / 2.0
            total_loss += pair_loss
            valid_pairs += 1
        
        if valid_pairs > 0:
            return total_loss / valid_pairs
        else:
            return torch.tensor(0.0, device=pred['descriptors'][0].device, requires_grad=True)



class DispersityPeakLoss(object):
    """
    Dispersity Peak Loss as described in ALIKED paper.
    Maximizes scores precisely at keypoints by penalizing high scores away from keypoint centers.
    """
    
    def __init__(self, window_size: int = 5, scores_th: float = 0.1):
        """
        Args:
            window_size: Size of the patch around each keypoint (W in the paper)
            scores_th: Threshold for valid keypoints
        """
        super().__init__()
        self.window_size = window_size
        self.scores_th = scores_th
        self.half_window = window_size // 2
        
    def extract_patches_around_keypoints(self, score_map, keypoints):
        """
        Extract WxW patches around keypoints from score map.
        
        Args:
            score_map: (H, W) score map
            keypoints: (N, 2) keypoint normalized coordinates [-1, 1]
            
        Returns:
            patches: List of (W, W) patches
            valid_kpts: List of valid keypoint coordinates
        """
        H, W = score_map.shape
        # print("SCORE MAP SHAPE: ", score_map.shape)
        patches = []
        valid_kpts = []
        
        for kpt in keypoints:
            # x, y = int(kpt[0]), int(kpt[1])
            x_norm, y_norm = kpt[0], kpt[1]
            x = int((x_norm + 1) * (W - 1) / 2)
            y = int((y_norm + 1) * (H - 1) / 2)
            # print(f"Processing keypoint at ({x}, {y})")

            # Check if patch is within bounds
            if (x - self.half_window >= 0 and x + self.half_window < W and
                y - self.half_window >= 0 and y + self.half_window < H):
                # print(f"Keypoint ({x}, {y}) is valid, extracting patch.")
                # Extract patch
                patch = score_map[y - self.half_window:y + self.half_window + 1,
                                x - self.half_window:x + self.half_window + 1]
                patches.append(patch)
                valid_kpts.append([x, y])
        # print(f"Extracted {len(patches)} patches around keypoints.")
        # print(f"Valid keypoints: {len(valid_kpts)}")     
        return patches, valid_kpts
    
    def compute_patch_loss(self, patch, keypoint_center):
        """
        Compute dispersity peak loss for a single patch.
        
        Args:
            patch: (W, W) score patch
            keypoint_center: [x, y] center coordinates relative to patch
            
        Returns:
            loss: scalar loss value
        """
        W = patch.shape[0]
        
        # Create coordinate grid for the patch
        y_coords, x_coords = torch.meshgrid(
            torch.arange(W, device=patch.device),
            torch.arange(W, device=patch.device),
            indexing='ij'
        )
        
        # Center coordinates (relative to patch)
        center_x, center_y = self.half_window, self.half_window
        
        # Calculate distances from each patch coordinate to keypoint center
        distances = torch.sqrt((x_coords - center_x)**2 + (y_coords - center_y)**2)
        
        # Apply softmax to patch scores
        patch_flat = patch.flatten()
        softmax_scores = F.softmax(patch_flat, dim=0)
        
        # Compute dot product: softmax(sp) · ||p - c||
        distances_flat = distances.flatten()
        loss = torch.mean(softmax_scores * distances_flat)
        
        return loss
    
    def __call__(self, pred):
        """
        Compute dispersity peak loss for all keypoints in batch.
        
        Args:
            pred: Dictionary containing:
                - 'scores_map': (B, C, H, W) score maps
                - 'scores': List of (N,) keypoint scores for each batch
                - 'keypoints': List of (N, 2) keypoint coordinates for each batch
                
        Returns:
            loss: Average dispersity peak loss
        """
        batch_size = pred['scores_map'].shape[0]
        total_loss = 0
        total_count = 0
        
        for batch_idx in range(batch_size):
            # Get score map for this batch (assuming single channel)
            score_map = pred['scores_map'][batch_idx, 0]  # (H, W)
            # print("SCORE MAP SHAPE: ", score_map.shape)
            # Get keypoints and scores
            if 'keypoints' in pred:
                keypoints = pred['keypoints'][batch_idx]  # (N, 2)
                scores = pred['scores'][batch_idx]# (N,)
                # print(f"Batch {batch_idx}: {len(keypoints)} keypoints")
                # print("KEYPOINTS SHAPE: ", keypoints.shape)
                # print("KEYPOINT SCORES  SHAPE:", scores.shape)
                # print("ALL SCORES: ", pred['scores'][batch_idx].shape)

            else:
                # If keypoints not provided, you might need to extract them
                # from the score map or use alternative method
                continue
            
            # Filter keypoints by score threshold
            # print(f"Batch {batch_idx}: Filtering keypoints with scores > {self.scores_th}")
            
            valid_mask = scores > self.scores_th

            if not valid_mask.any():
                continue
                
            valid_keypoints = keypoints[valid_mask]
            # print(f"Batch {batch_idx}: {len(valid_keypoints)} valid keypoints after thresholding")
            # print("VALID KEYPOINTS:", valid_keypoints)
            # Extract patches around valid keypoints
            patches, valid_kpt_coords = self.extract_patches_around_keypoints(
                score_map, valid_keypoints
            )
            
            # Compute loss for each patch
            batch_loss = 0
            for patch, kpt_coord in zip(patches, valid_kpt_coords):
                patch_loss = self.compute_patch_loss(patch, kpt_coord)
                batch_loss += patch_loss
                
            if len(patches) > 0:
                total_loss += batch_loss
                total_count += len(patches)
        
        # Return average loss
        if total_count > 0:
            return total_loss / total_count
        else:
            return pred['scores_map'].new_tensor(0.0, requires_grad=True)
        





class ReprojectionLocLoss(object):
    """
    Reprojection location errors of keypoints to train repeatable detector.
    """

    def __init__(self, norm: int = 1, scores_th: float = 0.1):
        super().__init__()
        self.norm = norm
        self.scores_th = scores_th

    def __call__(self, pred0, pred1, correspondences):
        b, c, h, w = pred0['scores_map'].shape
        loss_mean = 0
        CNT = 0
        for idx in range(b):
            if correspondences[idx]['correspondence0'] is None:
                continue

            if self.norm == 2:
                dist = correspondences[idx]['dist']
            elif self.norm == 1:
                dist = correspondences[idx]['dist_l1']
            else:
                raise TypeError('No such norm in correspondence.')

            ids0_d = correspondences[idx]['ids0_d']
            ids1_d = correspondences[idx]['ids1_d']

            scores0 = correspondences[idx]['scores0'].detach()[ids0_d]
            scores1 = correspondences[idx]['scores1'].detach()[ids1_d]
            valid = (scores0 > self.scores_th) * (scores1 > self.scores_th)
            # print("VALID scores " , valid.sum().item(), " out of ", len(valid))

            reprojection_errors = dist[ids0_d, ids1_d][valid]

            loss_mean = loss_mean + reprojection_errors.sum()
            CNT = CNT + len(reprojection_errors)

        loss_mean = loss_mean / CNT if CNT != 0 else correspondences[0]['dist'].new_tensor(0)

        assert not torch.isnan(loss_mean)
        return loss_mean


def local_similarity(descriptor_map, descriptors, kpts_wh, radius):
    """
    :param descriptor_map: CxHxW
    :param descriptors: NxC
    :param kpts_wh: Nx2 (W,H)
    :return:
    """
    _, h, w = descriptor_map.shape
    ksize = 2 * radius + 1

    descriptor_map_unflod = torch.nn.functional.unfold(descriptor_map.unsqueeze(0),
                                                       kernel_size=(ksize, ksize),
                                                       padding=(radius, radius))
    descriptor_map_unflod = descriptor_map_unflod[0].t().reshape(h * w, -1, ksize * ksize)
    # find the correspondence patch
    kpts_wh_long = kpts_wh.detach().long()
    patch_ids = kpts_wh_long[:, 0] + kpts_wh_long[:, 1] * h
    desc_patches = descriptor_map_unflod[patch_ids].permute(0, 2, 1).detach()  # N_kpts x s*s x 128

    local_sim = torch.einsum('nsd,nd->ns', desc_patches, descriptors)
    local_sim_sort = torch.sort(local_sim, dim=1, descending=True).values
    local_sim_sort_mean = local_sim_sort[:, 4:].mean(dim=1)  # 4 is safe radius for bilinear interplation

    return local_sim_sort_mean


class ScoreMapRepLoss(object):
    """ Scoremap repetability"""

    def __init__(self, temperature: float = 1):
        super().__init__()
        self.temperature = temperature
        self.radius = 2

    def __call__(self, pred0, pred1, correspondences):
        b, c, h, w = pred0['scores_map'].shape
        wh = pred0['keypoints'][0].new_tensor([[w - 1, h - 1]])
        loss_mean = 0
        CNT = 0

        for idx in range(b):
            if correspondences[idx]['correspondence0'] is None:
                continue

            scores_map0 = pred0['scores_map'][idx]
            scores_map1 = pred1['scores_map'][idx]
            kpts01 = correspondences[idx]['kpts01']
            kpts10 = correspondences[idx]['kpts10']  # valid warped keypoints

            # =====================
            scores_kpts10 = torch.nn.functional.grid_sample(scores_map0.unsqueeze(0), kpts10.view(1, 1, -1, 2),
                                                            mode='bilinear', align_corners=True)[0, 0, 0, :]
            scores_kpts01 = torch.nn.functional.grid_sample(scores_map1.unsqueeze(0), kpts01.view(1, 1, -1, 2),
                                                            mode='bilinear', align_corners=True)[0, 0, 0, :]

            s0 = scores_kpts01 * correspondences[idx]['scores0']  # repeatability
            s1 = scores_kpts10 * correspondences[idx]['scores1']  # repeatability

            # ===================== repetability
            similarity_map_01 = correspondences[idx]['similarity_map_01_valid']
            similarity_map_10 = correspondences[idx]['similarity_map_10_valid']

            pmf01 = ((similarity_map_01.detach() - 1) / self.temperature).exp()
            pmf10 = ((similarity_map_10.detach() - 1) / self.temperature).exp()

            kpts01 = kpts01.detach()
            kpts10 = kpts10.detach()

            pmf01_kpts = torch.nn.functional.grid_sample(pmf01.unsqueeze(0), kpts01.view(1, 1, -1, 2),
                                                         mode='bilinear', align_corners=True)[0, :, 0, :]
            pmf10_kpts = torch.nn.functional.grid_sample(pmf10.unsqueeze(0), kpts10.view(1, 1, -1, 2),
                                                         mode='bilinear', align_corners=True)[0, :, 0, :]
            repetability01 = torch.diag(pmf01_kpts)
            repetability10 = torch.diag(pmf10_kpts)

            # ===================== reliability
            # ids0, ids1 = correspondences[idx]['ids0'], correspondences[idx]['ids1']
            # descriptor_map0 = pred0['descriptor_map'][idx].detach()
            # descriptor_map1 = pred1['descriptor_map'][idx].detach()
            # descriptors0 = pred0['descriptors'][idx][ids0].detach()
            # descriptors1 = pred1['descriptors'][idx][ids1].detach()
            # kpts0 = pred0['keypoints'][idx][ids0].detach()
            # kpts1 = pred1['keypoints'][idx][ids1].detach()
            # kpts0_wh = (kpts0 / 2 + 0.5) * wh
            # kpts1_wh = (kpts1 / 2 + 0.5) * wh
            # ls0 = local_similarity(descriptor_map0, descriptors0, kpts0_wh, self.radius)
            # ls1 = local_similarity(descriptor_map1, descriptors1, kpts1_wh, self.radius)
            # reliability0 = 1 - ((ls0 - 1) / self.temperature).exp()
            # reliability1 = 1 - ((ls1 - 1) / self.temperature).exp()

            fs0 = repetability01  # * reliability0
            fs1 = repetability10  # * reliability1

            if s0.sum() != 0:
                loss01 = (1 - fs0) * s0 * len(s0) / s0.sum()
                loss_mean = loss_mean + loss01.sum()
                CNT = CNT + len(loss01)
            if s1.sum() != 0:
                loss10 = (1 - fs1) * s1 * len(s1) / s1.sum()
                loss_mean = loss_mean + loss10.sum()
                CNT = CNT + len(loss10)

        loss_mean = loss_mean / CNT if CNT != 0 else pred0['scores_map'].new_tensor(0)
        assert not torch.isnan(loss_mean)
        return loss_mean


class DescReprojectionLoss(object):
    """ Reprojection loss between warp and descriptor matching """

    def __init__(self, temperature=0.1):
        super().__init__()
        self.inv_temp = 1. / temperature

    def __call__(self, pred0, pred1, correspondences):
        b, c, h, w = pred0['scores_map'].shape
        device = pred0['scores_map'].device
        wh = pred0['keypoints'][0].new_tensor([[w - 1, h - 1]])
        loss_mean = 0
        CNT = 0

        for idx in range(b):
            if correspondences[idx]['correspondence0'] is None:
                continue

            kpts01, kpts10 = correspondences[idx]['kpts01'], correspondences[idx]['kpts10']  # valid warped keypoints

            similarity_map_01 = correspondences[idx]['similarity_map_01']
            similarity_map_10 = correspondences[idx]['similarity_map_10']
            ids0, ids1 = correspondences[idx]['ids0'], correspondences[idx]['ids1']
            ids0_out, ids1_out = correspondences[idx]['ids0_out'], correspondences[idx]['ids1_out']

            # ======================= valid
            similarity_map_01_valid, similarity_map_10_valid = similarity_map_01[ids0], similarity_map_10[ids1]
            similarity_map_01_valid = (similarity_map_01_valid - 1) * self.inv_temp
            similarity_map_10_valid = (similarity_map_10_valid - 1) * self.inv_temp

            # matching probability mass function
            pmf01_valid = torch.softmax(similarity_map_01_valid.view(-1, h * w), dim=1).view(-1, h, w)
            pmf10_valid = torch.softmax(similarity_map_10_valid.view(-1, h * w), dim=1).view(-1, h, w)

            pmf01_kpts_valid = torch.nn.functional.grid_sample(pmf01_valid.unsqueeze(0), kpts01.view(1, 1, -1, 2),
                                                               mode='bilinear', align_corners=True)[0, :, 0, :]
            pmf10_kpts_valid = torch.nn.functional.grid_sample(pmf10_valid.unsqueeze(0), kpts10.view(1, 1, -1, 2),
                                                               mode='bilinear', align_corners=True)[0, :, 0, :]
            # as we use the gt correspondence here, the outlier uniform pmf is ignored
            # C_{Q,N} in NRE
            C01 = torch.diag(pmf01_kpts_valid)
            C10 = torch.diag(pmf10_kpts_valid)

            # ======================= out
            similarity_map_01_out, similarity_map_10_out = similarity_map_01[ids0_out], similarity_map_10[ids1_out]
            out0 = torch.ones(len(similarity_map_01_out), device=device)
            out1 = torch.ones(len(similarity_map_10_out), device=device)
            # cat outside scores to similarity_map, thus similarity_map is (N, H*W +1)
            similarity_map_01_out = torch.cat([similarity_map_01_out.reshape(-1, h * w), out0[:, None]], dim=1)
            similarity_map_10_out = torch.cat([similarity_map_10_out.reshape(-1, h * w), out1[:, None]], dim=1)
            similarity_map_01_out = (similarity_map_01_out - 1) * self.inv_temp
            similarity_map_10_out = (similarity_map_10_out - 1) * self.inv_temp
            pmf01_out = torch.softmax(similarity_map_01_out, dim=1)
            pmf10_out = torch.softmax(similarity_map_10_out, dim=1)
            if len(pmf01_out) > 0:
                C01_out = pmf01_out[:, -1]
            else:
                C01_out = C01.new_tensor([])
            if len(pmf10_out) > 0:
                C10_out = pmf10_out[:, -1]
            else:
                C10_out = C10.new_tensor([])

            # ======================= out
            C = torch.cat([C01, C10, C01_out, C10_out])  # C
            C_widetilde = -C.log()  # \widetilde{C}

            loss_mean = loss_mean + C_widetilde.sum()
            CNT = CNT + len(C_widetilde)

        loss_mean = loss_mean / CNT if CNT != 0 else wh.new_tensor(0)
        assert not torch.isnan(loss_mean)
        return loss_mean

