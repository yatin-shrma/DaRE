from collections import Counter
from decisionTree import DecisionTree
import numpy as np

class RandomForest:
    def __init__(self, n_trees=10, max_depth=10, min_sample_split=2, n_features=None, *, dr_max=0, k=1):
        self.n_trees = n_trees
        self.max_depth = max_depth
        self.min_sample_split = min_sample_split
        self.n_features = n_features
        self.trees = []
        self.k = k
        self.dr_max = dr_max
        
    def fit(self, X, y):
        for _ in range(self.n_trees):
            tree = DecisionTree(
                max_depth=self.max_depth,
                min_sample_split=self.min_sample_split,
                n_features=self.n_features,
                dr_max=self.dr_max,
                k=self.k
            )
            
            X_sample, y_sample = self._bootstrap_samples(X, y)
            tree.fit(X_sample, y_sample)
            self.trees.append(tree)
            
    def _bootstrap_samples(self, X, y):
        n_samples = X.shape[0]
        idx = np.random.choice(n_samples, n_samples, replace=True)
        return X[idx], y[idx]
        
    def _most_common_label(self, y):
        counter = Counter(y)
        return counter.most_common(1)[0][0]
        
    def predict(self, X):
        predictions = np.array([tree.predict(X) for tree in self.trees])
        tree_preds = np.swapaxes(predictions, 0, 1)
        predictions = np.array([self._most_common_label(pred) for pred in tree_preds])
        return predictions
        
    def accuracy(self, Y_pred, Y_test):
        return np.sum(Y_pred == Y_test) / len(Y_test)
        
    def unlearn(self, x, y):
        """Remove a data point from all trees in the forest"""
        for tree in self.trees:
            tree.unlearn(x, y)
            
    def feature_importance(self):
        """Calculate feature importance across all trees"""
        if not self.trees:
            return None
            
        n_features = self.trees[0].n_features
        importances = np.zeros(n_features)
        
        # Count feature usage at non-leaf nodes as a measure of importance
        for tree in self.trees:
            def count_feature_usage(node):
                if node is None or node.is_leaf_node():
                    return {}
                    
                counts = {node.feature: 1}
                
                # Recursively count in subtrees
                left_counts = count_feature_usage(node.left)
                right_counts = count_feature_usage(node.right)
                
                # Combine counts
                for feat, count in left_counts.items():
                    counts[feat] = counts.get(feat, 0) + count
                
                for feat, count in right_counts.items():
                    counts[feat] = counts.get(feat, 0) + count
                    
                return counts
            
            feature_counts = count_feature_usage(tree.root)
            for feat, count in feature_counts.items():
                if feat is not None and feat < n_features:
                    importances[feat] += count
        
        # Normalize
        if np.sum(importances) > 0:
            importances = importances / np.sum(importances)
            
        return importances
        
    def get_oob_score(self, X, y):
        """Calculate Out-of-Bag score"""
        n_samples = X.shape[0]
        oob_preds = np.zeros((n_samples, self.n_trees))
        n_oob_votes = np.zeros(n_samples)
        
        # For each tree, predict on samples that weren't used for training
        for i, tree in enumerate(self.trees):
            # Generate bootstrap indices
            n_samples = X.shape[0]
            bootstrap_indices = np.random.choice(n_samples, n_samples, replace=True)
            
            # OOB samples are those not in bootstrap sample
            oob_indices = np.array([i for i in range(n_samples) if i not in bootstrap_indices])
            
            if len(oob_indices) > 0:
                # Get predictions for OOB samples
                oob_predictions = tree.predict(X[oob_indices])
                
                # Store predictions
                for j, idx in enumerate(oob_indices):
                    oob_preds[idx, i] = oob_predictions[j]
                    n_oob_votes[idx] += 1
        
        # Aggregate predictions for each sample
        final_preds = []
        for i in range(n_samples):
            if n_oob_votes[i] > 0:
                # Get non-zero predictions for this sample
                sample_preds = oob_preds[i, oob_preds[i, :] > 0]
                if len(sample_preds) > 0:
                    # Mode of predictions
                    counter = Counter(sample_preds)
                    final_preds.append(counter.most_common(1)[0][0])
                else:
                    # Default prediction if no valid OOB predictions
                    final_preds.append(self._most_common_label(y))
            else:
                # No OOB predictions, use global most common
                final_preds.append(self._most_common_label(y))
        
        # Calculate accuracy
        return np.sum(np.array(final_preds) == y) / len(y)