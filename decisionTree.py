import numpy as np
from collections import Counter

class Stats:
    def __init__(self, s_leftp, s_lT, s_right, s_rT):
        self.s_left = s_leftp    # Left positive instances count
        self.s_lT = s_lT         # Left total instances count
        self.s_right = s_right   # Right positive instances count
        self.s_rT = s_rT         # Right total instances count

class Node:
    def __init__(self, feature=None, left=None, right=None, threshold=None, 
                 n_Dnode=None, n_D1=None, *, value=None, n_type='G', itr=0):
        self.feature = feature
        self.left = left
        self.right = right
        self.threshold = threshold
        self.value = value
        self.n_type = n_type      # 'G' for greedy, 'R' for random, 'L' for leaf
        
        # DaRE specific attributes
        self.itr = itr            # Current threshold index
        self.n_Dnode = n_Dnode    # Total instances at node
        self.n_D1 = n_D1          # Positive instances at node
        self.k_holds = np.array([])  # Stored threshold candidates
        self.res_feat = np.array([])  # Stored feature candidates
        self.stats = []           # Statistics for each candidate split
        self.ig = 0               # Information gain
        self.X = None             # Training data at this node (for leaf nodes)
        self.Y = None             # Labels at this node (for leaf nodes)
        
    def is_leaf_node(self):
        return self.value is not None

class DecisionTree:
    def __init__(self, min_sample_split=2, max_depth=10, n_features=None, dr_max=0, k=1):
        self.min_sample_split = min_sample_split
        self.max_depth = max_depth
        self.n_features = n_features  # Number of features to consider for splits
        self.root = None
        self.dr_max = dr_max      # Maximum depth for random nodes
        self.k = k                # Number of threshold candidates to maintain

    def fit(self, X, y):
        # Ensure n_features is properly set - fix: handle when n_features is None
        if self.n_features is None:
            self.n_features = X.shape[1]
        else:
            self.n_features = min(X.shape[1], self.n_features)
            
        self.root = self._grow_tree(X, y)

    def _grow_tree(self, X, y, depth=0):
        # Evaluating the stopping criteria
        n_samples, n_feats = X.shape
        n_label = len(np.unique(y))

        if (depth >= self.max_depth or self.min_sample_split > n_samples or n_label == 1):
            leaf_value = self._most_common_label(y)
            dnode = len(y)
            pos = np.sum(y)
            newNode = Node(value=leaf_value, n_Dnode=dnode, n_D1=pos, n_type='L')
            # Store data for potential unlearning
            newNode.X = X.copy()
            newNode.Y = y.copy()
            return newNode
        
        # Random node selection for top dr_max levels
        if depth < self.dr_max:
            # Fix: Ensure n_feats is not zero before choosing random feature
            if n_feats == 0:
                leaf_value = self._most_common_label(y)
                return Node(value=leaf_value, n_Dnode=len(y), n_D1=np.sum(y), n_type='L')
                
            # Select random feature
            random_feat = np.random.choice(n_feats, 1)[0]
            
            # Find feature bounds
            thresh_bound_L = np.min(X[:, random_feat])
            thresh_bound_U = np.max(X[:, random_feat])
            
            # Generate random threshold within bounds
            random_thresh = thresh_bound_L + (np.random.random()*0.8 * (thresh_bound_U - thresh_bound_L))
            # random_thresh = thresh_bound_L + (0.5 * (thresh_bound_U - thresh_bound_L))
            # Split data
            left_idx, right_idx = self._split(X[:, random_feat], random_thresh)
            
            # Handle edge case: if split results in empty node, adjust threshold
            if len(left_idx) == 0 or len(right_idx) == 0:
                # Choose a threshold that results in balanced split
                sorted_vals = np.sort(X[:, random_feat])
                mid_idx = len(sorted_vals) // 2
                random_thresh = sorted_vals[mid_idx]
                left_idx, right_idx = self._split(X[:, random_feat], random_thresh)
            
            # Recursively grow subtrees
            left = self._grow_tree(X[left_idx, :], y[left_idx], depth+1)
            right = self._grow_tree(X[right_idx, :], y[right_idx], depth+1)

            # Create node and store statistics
            dnode = len(y)
            pos = np.sum(y)
            newNode = Node(random_feat, left, right, random_thresh, n_Dnode=dnode, n_D1=pos, n_type='R')
            
            # Store statistics for potential unlearning
            left_pos = np.sum(y[left_idx])
            right_pos = np.sum(y[right_idx])
            stat = Stats(s_leftp=left_pos, s_lT=len(left_idx), s_right=right_pos, s_rT=len(right_idx))
            newNode.stats = [stat]
            return newNode

        else:
            # Fix: Ensure n_feats is not zero before feature selection
            if n_feats == 0:
                leaf_value = self._most_common_label(y)
                return Node(value=leaf_value, n_Dnode=len(y), n_D1=np.sum(y), n_type='L')
            
            # Feature selection for greedy nodes - ensure we don't select more than available
            n_features_to_sample = min(self.n_features, n_feats)
            if n_features_to_sample <= 0:
                # If no features to sample, create a leaf node
                leaf_value = self._most_common_label(y)
                return Node(value=leaf_value, n_Dnode=len(y), n_D1=np.sum(y), n_type='L')
                
            feat_ids = np.random.choice(n_feats, n_features_to_sample, replace=False)

            # Find best split among selected features
            best_thresh, best_feat, threshK, itr, featK, gain = self._best_split(X, y, feat_ids)
            
            # Fix: If no valid split was found, create a leaf node
            if best_thresh is None or best_feat is None:
                leaf_value = self._most_common_label(y)
                dnode = len(y)
                pos = np.sum(y)
                newNode = Node(value=leaf_value, n_Dnode=dnode, n_D1=pos, n_type='L')
                newNode.X = X.copy()
                newNode.Y = y.copy()
                return newNode
        
            # Split data
            left_idx, right_idx = self._split(X[:, best_feat], best_thresh)

            # Recursively grow subtrees
            left = self._grow_tree(X[left_idx, :], y[left_idx], depth+1)
            right = self._grow_tree(X[right_idx, :], y[right_idx], depth+1)

            # Create node and store statistics
            dnode = len(y)
            pos = np.sum(y)
            newNode = Node(best_feat, left, right, best_thresh, n_Dnode=dnode, n_D1=pos)
            newNode.k_holds = threshK
            newNode.res_feat = featK
            newNode.stats = self._split_for_stats(threshK, featK, X, y)
            newNode.itr = itr
            newNode.ig = gain
            return newNode

    def _split_for_stats(self, threshK, featK, X, y):
        """Create statistics for all threshold candidates"""
        lenT = len(threshK)
        ret_stat = []
        for i in range(lenT):
            feat = featK[i]
            thr = threshK[i]
            # Fix: Handle the case where feat might be None
            if feat is None or feat >= X.shape[1]:
                continue
            
            left, right = self._split(X[:, feat], thr)
            left_pos = np.sum(y[left]) if len(left) > 0 else 0
            right_pos = np.sum(y[right]) if len(right) > 0 else 0
            stat = Stats(s_leftp=left_pos, s_lT=len(left), s_right=right_pos, s_rT=len(right))
            ret_stat.append(stat)
        return ret_stat
    
    def _best_split(self, X, y, feat_ids):
        """Find best split and maintain k alternative candidates"""
        best_gain = -1
        split_thresh, split_id = None, None
        
        # Initialize arrays for k best thresholds and features
        kh = np.zeros(self.k)  # Thresholds
        kf = np.zeros(self.k, dtype=int)  # Features
        gains = np.zeros(self.k) - 1  # Information gains
        
        itr = 0  # Index for next replacement
        
        for feat_id in feat_ids:
            X_colmn = X[:, feat_id]
            thresholds = np.unique(X_colmn)
            
            # Skip features with only one unique value
            if len(thresholds) <= 1:
                continue
                
            for thr in thresholds:
                # Calculate information gain for this split
                cur_gain = self._information_gain(X_colmn, y, thr)
                
                # Update best split if better than current best
                if best_gain < cur_gain:
                    best_gain = cur_gain
                    split_id = feat_id
                    split_thresh = thr
                
                # Update k-best list if better than any current candidate
                min_gain_idx = np.argmin(gains)
                if cur_gain > gains[min_gain_idx]:
                    gains[min_gain_idx] = cur_gain
                    kh[min_gain_idx] = thr
                    kf[min_gain_idx] = feat_id
                    itr = min_gain_idx  # Mark this slot as most recently updated

        # If no valid split found
        if split_thresh is None:
            # Choose most common value
            leaf_value = self._most_common_label(y)
            return None, None, kh, itr, kf, best_gain
            
        return split_thresh, split_id, kh, itr, kf, best_gain

    def _information_gain(self, X, y, thr):
        """Calculate information gain for a split"""
        parent_entropy = self._entropy(y)

        # Split data
        left_idx, right_idx = self._split(X, thr)
        
        # Skip invalid splits
        if len(left_idx) == 0 or len(right_idx) == 0:
            return 0

        # Calculate weighted entropy of children
        n = len(y)
        n_l, n_r = len(left_idx), len(right_idx)
        e_l = self._entropy(y[left_idx])
        e_r = self._entropy(y[right_idx])
        
        child_entropy = (n_l/n) * e_l + (n_r/n) * e_r

        return parent_entropy - child_entropy

    def _split(self, X, thr):
        """Split data based on threshold"""
        # Fix: Handle None threshold case
        if thr is None:
            # Return all data to one side if threshold is None
            return np.array([], dtype=int), np.arange(len(X), dtype=int)
            
        left_idx = np.argwhere(X <= thr).flatten()
        right_idx = np.argwhere(X > thr).flatten()
        return left_idx, right_idx

    def _entropy(self, y):
        """Calculate entropy of a label distribution"""
        # Handle empty array
        if len(y) == 0:
            return 0
            
        hist = np.bincount(y)
        ps = hist / len(y)
        
        # Only include non-zero probabilities
        return -np.sum([p * np.log2(p) for p in ps if p > 0])

    def _most_common_label(self, y):
        """Find most common label in an array"""
        if len(y) == 0:
            return 0  # Default value for empty array
            
        counter = Counter(y)
        return counter.most_common(1)[0][0]

    def predict(self, X):
        """Predict class for samples in X"""
        return np.array([self._traverse_tree(x, self.root) for x in X])

    def _traverse_tree(self, x, node):
        """Traverse tree to find prediction for a single sample"""
        if node.is_leaf_node():
            return node.value
            
        if x[node.feature] <= node.threshold:
            return self._traverse_tree(x, node.left)
        else:
            return self._traverse_tree(x, node.right)

    def accuracy(self, Y_pred, Y_test):
        """Calculate accuracy of predictions"""
        return np.sum(Y_pred == Y_test) / len(Y_test)

    def unlearn(self, x, y):
        """Remove a data point from the model"""
        self.find_and_unlearn(x, y, self.root)
        
    def _fetch_data_without_point(self, node, x):
        """Get all data at this node except the point to be unlearned"""
        if node is None:
            return np.array([]), np.array([])
            
        if node.n_type == 'L':
            # For leaf nodes, we have the data stored
            if node.X is None or len(node.X) == 0:
                return np.array([]), np.array([])
                
            # Find instances that don't match x
            matches = np.array([not np.array_equal(row, x) for row in node.X])
            if np.any(matches):
                return node.X[matches], node.Y[matches]
            return np.array([]), np.array([])
            
        # For internal nodes, combine data from subtrees
        left_X, left_Y = self._fetch_data_without_point(node.left, x)
        right_X, right_Y = self._fetch_data_without_point(node.right, x)
        
        # Combine data
        if len(left_X) == 0:
            return right_X, right_Y
        elif len(right_X) == 0:
            return left_X, left_Y
        else:
            return np.vstack([left_X, right_X]), np.hstack([left_Y, right_Y])

    def find_and_unlearn(self, x, y, node, depth=0):
        """Recursively find and unlearn a data point"""
        if node is None:
            return
            
        # Update instance counts
        node.n_Dnode -= 1
        if y == 1:
            node.n_D1 -= 1
            
        need_rebuild = False
        
        # Handle different node types
        if node.n_type == 'L':
            # For leaf nodes, just remove the point from stored data
            if node.X is not None and len(node.X) > 0:
                matches = np.array([not np.array_equal(row, x) for row in node.X])
                node.X = node.X[matches] if np.any(matches) else np.array([])
                node.Y = node.Y[matches] if np.any(matches) else np.array([])
            return
            
        elif node.n_type == 'R':
            # For random nodes, update statistics
            if x[node.feature] <= node.threshold:
                node.stats[0].s_lT -= 1
                if y == 1:
                    node.stats[0].s_left -= 1
            else:
                node.stats[0].s_rT -= 1  # Fixed: was incorrectly setting to -1
                if y == 1:
                    node.stats[0].s_right -= 1
            
            # Check if split balance is still acceptable
            sr_t = max(1, node.stats[0].s_rT)  # Avoid division by zero
            sl_t = max(1, node.stats[0].s_lT)  # Avoid division by zero
            
            ratio = sl_t / sr_t
            
            # If imbalanced, rebuild the node
            if ratio <= 0.2 or ratio >= 5.0:
                need_rebuild = True
                
        elif node.n_type == 'G':
            # For greedy nodes, update all candidate statistics and check if best split changes
            need_rebuild, new_thr, new_feat = self._update_threshold_stats(x, y, node)
            
            if need_rebuild:
                node.threshold = new_thr
                node.feature = new_feat
                
        # If node needs rebuilding
        if need_rebuild:
            # Get remaining data
            X, Y = self._fetch_data_without_point(node, x)
            
            if len(X) > 0:
                # Rebuild the subtree
                new_node = self._grow_tree(X, Y, depth)
                # Copy properties to current node
                node.feature = new_node.feature
                node.threshold = new_node.threshold
                node.value = new_node.value
                node.left = new_node.left
                node.right = new_node.right
                node.n_type = new_node.n_type
                node.k_holds = new_node.k_holds
                node.res_feat = new_node.res_feat
                node.stats = new_node.stats
                node.itr = new_node.itr
                node.ig = new_node.ig
                node.X = new_node.X
                node.Y = new_node.Y
            else:
                # No data left, make it a leaf node
                node.value = 0  # Default value 
                node.left = None
                node.right = None
                node.n_type = 'L'
                
        else:
            # Continue down the tree
            if node.feature is None:
                # If feature is None, we can't traverse further
                return
                
            if x[node.feature] <= node.threshold:
                self.find_and_unlearn(x, y, node.left, depth+1)
            else:
                self.find_and_unlearn(x, y, node.right, depth+1)

    def _update_threshold_stats(self, x, y, node):
        """Update statistics for all threshold candidates and check if best split changes"""
        best_ig = -1
        best_thr = None
        best_feat = None
        need_rebuild = False
        
        # Update statistics for each candidate
        for i in range(min(self.k, len(node.k_holds))):
            cur_i = i
            feat = int(node.res_feat[cur_i])
            thr = node.k_holds[cur_i]
            
            # Update the appropriate statistics
            if x[feat] <= thr:
                node.stats[cur_i].s_lT -= 1
                if y == 1:
                    node.stats[cur_i].s_left -= 1
            else:
                node.stats[cur_i].s_rT -= 1  # Fixed: was incorrectly setting to -1
                if y == 1:
                    node.stats[cur_i].s_right -= 1
                    
            # Skip invalid splits
            if node.stats[cur_i].s_lT <= 0 or node.stats[cur_i].s_rT <= 0:
                continue
                
            # Check split balance
            y_right = node.stats[cur_i].s_rT
            y_left = node.stats[cur_i].s_lT
            
            ratio = y_left / y_right
            if ratio <= 0.2 or ratio >= 5.0:
                need_rebuild = True
                
            # Calculate entropy and information gain
            try:
                y_rpos = max(0, min(node.stats[cur_i].s_right, y_right))  # Ensure valid range
                y_lpos = max(0, min(node.stats[cur_i].s_left, y_left))    # Ensure valid range
                
                # Calculate parent entropy
                total = y_left + y_right
                total_pos = y_lpos + y_rpos
                p_pos = total_pos / total
                p_neg = 1 - p_pos
                
                if 0 < p_pos < 1:  # Only calculate if not pure
                    H_p = -(p_pos * np.log2(p_pos)) - (p_neg * np.log2(p_neg))
                else:
                    H_p = 0
                
                # Calculate child entropies
                h_right = 0
                if y_right > 0:
                    p_rpos = y_rpos / y_right
                    if 0 < p_rpos < 1:
                        h_right = -(p_rpos * np.log2(p_rpos)) - ((1-p_rpos) * np.log2(1-p_rpos))
                        
                h_left = 0
                if y_left > 0:
                    p_lpos = y_lpos / y_left
                    if 0 < p_lpos < 1:
                        h_left = -(p_lpos * np.log2(p_lpos)) - ((1-p_lpos) * np.log2(1-p_lpos))
                
                # Calculate weighted entropy and information gain
                h_child = ((y_right/total) * h_right) + ((y_left/total) * h_left)
                i_g = H_p - h_child
                
                # Track best split
                if i_g > best_ig:
                    best_ig = i_g
                    best_thr = thr
                    best_feat = feat
                    
            except (ZeroDivisionError, ValueError):
                # Skip invalid calculations
                continue
                
        # Check if best split has changed
        if best_feat is not None and best_thr is not None:
            if node.feature != best_feat or node.threshold != best_thr:
                need_rebuild = True
                
        return need_rebuild, best_thr, best_feat