"""
Siamese RNN Model with Triplet Loss - PyTorch Implementation
Advanced keystroke dynamics authentication using metric learning
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pickle
import os
from tqdm import tqdm


class KeystrokeEncoder(nn.Module):
    """
    LSTM-based encoder for keystroke sequences.
    
    Architecture:
    Input → LSTM(128) → Dropout → LSTM(64) → Dropout → Dense(32) → L2 Normalize
    """
    
    def __init__(self, input_dim=4, hidden_dim1=128, hidden_dim2=64, embedding_dim=32):
        """
        Initialize the encoder network.
        
        Args:
            input_dim: Number of features per timestep (default: 4)
            hidden_dim1: Size of first LSTM layer (default: 128)
            hidden_dim2: Size of second LSTM layer (default: 64)
            embedding_dim: Size of final embedding (default: 32)
        """
        super(KeystrokeEncoder, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_dim1 = hidden_dim1
        self.hidden_dim2 = hidden_dim2
        self.embedding_dim = embedding_dim
        
        # First LSTM layer
        self.lstm1 = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim1,
            num_layers=1,
            batch_first=True,
            bidirectional=False
        )
        
        # Batch normalization after first LSTM
        self.bn1 = nn.BatchNorm1d(hidden_dim1)
        
        # Dropout for regularization
        self.dropout1 = nn.Dropout(0.3)
        
        # Second LSTM layer
        self.lstm2 = nn.LSTM(
            input_size=hidden_dim1,
            hidden_size=hidden_dim2,
            num_layers=1,
            batch_first=True,
            bidirectional=False
        )
        
        # Batch normalization after second LSTM
        self.bn2 = nn.BatchNorm1d(hidden_dim2)
        
        # Dropout
        self.dropout2 = nn.Dropout(0.3)
        
        # Dense layer for embedding
        self.fc = nn.Linear(hidden_dim2, embedding_dim)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize network weights using Xavier initialization."""
        for name, param in self.named_parameters():
            if 'weight_ih' in name:
                nn.init.xavier_uniform_(param.data)
            elif 'weight_hh' in name:
                nn.init.orthogonal_(param.data)
            elif 'bias' in name:
                param.data.fill_(0)
            elif 'fc.weight' in name:
                nn.init.xavier_uniform_(param.data)
    
    def forward(self, x):
        """
        Forward pass through the encoder.
        
        Args:
            x: Input tensor of shape (batch, sequence_length, input_dim)
        
        Returns:
            embedding: L2-normalized embedding of shape (batch, embedding_dim)
        """
        batch_size = x.size(0)
        
        # First LSTM layer
        # Input: (batch, seq_len, input_dim)
        # Output: (batch, seq_len, hidden_dim1)
        lstm1_out, _ = self.lstm1(x)
        
        # Apply batch norm on last timestep
        # lstm1_out[:, -1, :] gives us (batch, hidden_dim1)
        last_hidden1 = lstm1_out[:, -1, :]
        last_hidden1 = self.bn1(last_hidden1)
        last_hidden1 = self.dropout1(last_hidden1)
        
        # Prepare for second LSTM: expand back to sequence
        # Shape: (batch, 1, hidden_dim1)
        lstm2_input = last_hidden1.unsqueeze(1)
        
        # Second LSTM layer
        lstm2_out, _ = self.lstm2(lstm2_input)
        
        # Get final hidden state
        last_hidden2 = lstm2_out[:, -1, :]
        last_hidden2 = self.bn2(last_hidden2)
        last_hidden2 = self.dropout2(last_hidden2)
        
        # Dense layer
        embedding = self.fc(last_hidden2)
        embedding = F.relu(embedding)
        
        # L2 normalization
        # Makes all embeddings lie on unit hypersphere
        embedding = F.normalize(embedding, p=2, dim=1)
        
        return embedding


class SiameseRNNTriplet(nn.Module):
    """
    Siamese Network using triplet loss for keystroke authentication.
    
    Uses shared encoder for anchor, positive, and negative samples.
    """
    
    def __init__(self, input_dim=4, hidden_dim1=128, hidden_dim2=64, embedding_dim=32):
        """
        Initialize Siamese network.
        
        Args:
            input_dim: Number of features per keystep
            hidden_dim1: First LSTM hidden size
            hidden_dim2: Second LSTM hidden size
            embedding_dim: Final embedding dimension
        """
        super(SiameseRNNTriplet, self).__init__()
        
        # Shared encoder for all three inputs
        self.encoder = KeystrokeEncoder(
            input_dim=input_dim,
            hidden_dim1=hidden_dim1,
            hidden_dim2=hidden_dim2,
            embedding_dim=embedding_dim
        )
        
        self.embedding_dim = embedding_dim
    
    def forward(self, anchor, positive, negative):
        """
        Forward pass for triplet.
        
        Args:
            anchor: Anchor samples (batch, seq_len, input_dim)
            positive: Positive samples (batch, seq_len, input_dim)
            negative: Negative samples (batch, seq_len, input_dim)
        
        Returns:
            anchor_embed: Anchor embeddings (batch, embedding_dim)
            positive_embed: Positive embeddings (batch, embedding_dim)
            negative_embed: Negative embeddings (batch, embedding_dim)
        """
        anchor_embed = self.encoder(anchor)
        positive_embed = self.encoder(positive)
        negative_embed = self.encoder(negative)
        
        return anchor_embed, positive_embed, negative_embed
    
    def get_embedding(self, x):
        """
        Get embedding for a single sample.
        
        Args:
            x: Input sample (batch, seq_len, input_dim)
        
        Returns:
            embedding: Embedding vector (batch, embedding_dim)
        """
        return self.encoder(x)


class TripletLoss(nn.Module):
    """
    Triplet Loss with margin.
    
    Formula: L = max(0, ||a-p||² - ||a-n||² + margin)
    """
    
    def __init__(self, margin=0.2):
        """
        Initialize triplet loss.
        
        Args:
            margin: Minimum separation between positive and negative
        """
        super(TripletLoss, self).__init__()
        self.margin = margin
    
    def forward(self, anchor, positive, negative):
        """
        Calculate triplet loss.
        
        Args:
            anchor: Anchor embeddings (batch, embedding_dim)
            positive: Positive embeddings (batch, embedding_dim)
            negative: Negative embeddings (batch, embedding_dim)
        
        Returns:
            loss: Scalar triplet loss
        """
        # Calculate squared Euclidean distances
        pos_dist = F.pairwise_distance(anchor, positive, p=2).pow(2)
        neg_dist = F.pairwise_distance(anchor, negative, p=2).pow(2)
        
        # Triplet loss with margin
        losses = F.relu(pos_dist - neg_dist + self.margin)
        
        return losses.mean()


class TripletDataset(Dataset):
    """
    PyTorch Dataset for triplet training.
    """
    
    def __init__(self, anchors, positives, negatives):
        """
        Initialize dataset.
        
        Args:
            anchors: Numpy array of anchor samples
            positives: Numpy array of positive samples
            negatives: Numpy array of negative samples
        """
        self.anchors = torch.FloatTensor(anchors)
        self.positives = torch.FloatTensor(positives)
        self.negatives = torch.FloatTensor(negatives)
        
        assert len(self.anchors) == len(self.positives) == len(self.negatives)
    
    def __len__(self):
        return len(self.anchors)
    
    def __getitem__(self, idx):
        return (
            self.anchors[idx],
            self.positives[idx],
            self.negatives[idx]
        )


class TripletGenerator:
    """
    Generate triplets from user samples for training.
    """
    
    @staticmethod
    def generate_triplets(user_samples, num_triplets=1000):
        """
        Generate random triplets from user samples.
        
        Args:
            user_samples: Dict mapping user_id to list of feature arrays
            num_triplets: Number of triplets to generate
        
        Returns:
            anchors, positives, negatives: Numpy arrays
        """
        anchors = []
        positives = []
        negatives = []
        
        user_ids = list(user_samples.keys())
        
        if len(user_ids) < 2:
            raise ValueError("Need at least 2 users to generate triplets")
        
        for _ in range(num_triplets):
            # Select random user for anchor and positive
            user_id = np.random.choice(user_ids)
            user_data = user_samples[user_id]
            
            # Need at least 2 samples from this user
            if len(user_data) < 2:
                continue
            
            # Random anchor and positive (same user, different samples)
            anchor_idx, positive_idx = np.random.choice(
                len(user_data),
                size=2,
                replace=False
            )
            
            # Random negative (different user)
            negative_user = np.random.choice(
                [uid for uid in user_ids if uid != user_id]
            )
            negative_idx = np.random.randint(len(user_samples[negative_user]))
            
            # Add to lists
            anchors.append(user_data[anchor_idx])
            positives.append(user_data[positive_idx])
            negatives.append(user_samples[negative_user][negative_idx])
        
        return np.array(anchors), np.array(positives), np.array(negatives)
    
    @staticmethod
    def generate_hard_triplets(model, user_samples, num_triplets=500, device='cpu'):
        """
        Generate hard triplets - challenging examples for training.
        
        Args:
            model: Trained SiameseRNNTriplet model
            user_samples: Dict mapping user_id to list of feature arrays
            num_triplets: Number of hard triplets to generate
            device: 'cpu' or 'cuda'
        
        Returns:
            anchors, positives, negatives: Numpy arrays
        """
        model.eval()
        anchors = []
        positives = []
        negatives = []
        
        user_ids = list(user_samples.keys())
        
        with torch.no_grad():
            for _ in range(num_triplets):
                # Select random user
                user_id = np.random.choice(user_ids)
                user_data = user_samples[user_id]
                
                if len(user_data) < 2:
                    continue
                
                # Random anchor
                anchor_idx = np.random.randint(len(user_data))
                anchor = user_data[anchor_idx]
                
                # Get anchor embedding
                anchor_tensor = torch.FloatTensor(anchor).unsqueeze(0).to(device)
                anchor_embedding = model.get_embedding(anchor_tensor).cpu().numpy()[0]
                
                # Find hardest positive (furthest same-user sample)
                max_dist = -1
                hard_positive_idx = 0
                for i, sample in enumerate(user_data):
                    if i == anchor_idx:
                        continue
                    sample_tensor = torch.FloatTensor(sample).unsqueeze(0).to(device)
                    sample_embedding = model.get_embedding(sample_tensor).cpu().numpy()[0]
                    dist = np.sum((anchor_embedding - sample_embedding) ** 2)
                    if dist > max_dist:
                        max_dist = dist
                        hard_positive_idx = i
                
                # Find hardest negative (closest different-user sample)
                min_dist = float('inf')
                hard_negative_user = None
                hard_negative_idx = 0
                
                for neg_user in user_ids:
                    if neg_user == user_id:
                        continue
                    for i, sample in enumerate(user_samples[neg_user]):
                        sample_tensor = torch.FloatTensor(sample).unsqueeze(0).to(device)
                        sample_embedding = model.get_embedding(sample_tensor).cpu().numpy()[0]
                        dist = np.sum((anchor_embedding - sample_embedding) ** 2)
                        if dist < min_dist:
                            min_dist = dist
                            hard_negative_user = neg_user
                            hard_negative_idx = i
                
                # Add hard triplet
                anchors.append(anchor)
                positives.append(user_data[hard_positive_idx])
                negatives.append(user_samples[hard_negative_user][hard_negative_idx])
        
        model.train()
        return np.array(anchors), np.array(positives), np.array(negatives)


class Trainer:
    """
    Training manager for Siamese Triplet Network.
    """
    
    def __init__(self, model, device='cpu', learning_rate=0.001):
        """
        Initialize trainer.
        
        Args:
            model: SiameseRNNTriplet model
            device: 'cpu' or 'cuda'
            learning_rate: Learning rate for optimizer
        """
        self.model = model.to(device)
        self.device = device
        self.criterion = TripletLoss(margin=0.2)
        self.optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=0.5,
            patience=5,
        )
    
    def train_epoch(self, dataloader):
        """
        Train for one epoch.
        
        Args:
            dataloader: PyTorch DataLoader with triplets
        
        Returns:
            avg_loss: Average loss for the epoch
        """
        self.model.train()
        total_loss = 0
        num_batches = 0
        
        for anchor, positive, negative in tqdm(dataloader, desc="Training"):
            # Move to device
            anchor = anchor.to(self.device)
            positive = positive.to(self.device)
            negative = negative.to(self.device)
            
            # Forward pass
            anchor_embed, positive_embed, negative_embed = self.model(
                anchor, positive, negative
            )
            
            # Calculate loss
            loss = self.criterion(anchor_embed, positive_embed, negative_embed)
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping to prevent exploding gradients
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            self.optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
        
        avg_loss = total_loss / num_batches
        return avg_loss
    
    def validate(self, dataloader):
        """
        Validate the model.
        
        Args:
            dataloader: Validation DataLoader
        
        Returns:
            avg_loss: Average validation loss
        """
        self.model.eval()
        total_loss = 0
        num_batches = 0
        
        with torch.no_grad():
            for anchor, positive, negative in dataloader:
                anchor = anchor.to(self.device)
                positive = positive.to(self.device)
                negative = negative.to(self.device)
                
                anchor_embed, positive_embed, negative_embed = self.model(
                    anchor, positive, negative
                )
                
                loss = self.criterion(anchor_embed, positive_embed, negative_embed)
                
                total_loss += loss.item()
                num_batches += 1
        
        avg_loss = total_loss / num_batches
        return avg_loss
    
    def train(self, train_loader, val_loader=None, epochs=50, save_path='best_model.pt'):
        """
        Complete training loop.
        
        Args:
            train_loader: Training DataLoader
            val_loader: Validation DataLoader (optional)
            epochs: Number of epochs
            save_path: Path to save best model
        
        Returns:
            history: Dict with training history
        """
        history = {
            'train_loss': [],
            'val_loss': []
        }
        
        best_val_loss = float('inf')
        patience_counter = 0
        patience = 10
        
        print(f"\nTraining on device: {self.device}")
        print(f"Total epochs: {epochs}\n")
        
        for epoch in range(epochs):
            print(f"Epoch {epoch + 1}/{epochs}")
            print("-" * 50)
            
            # Train
            train_loss = self.train_epoch(train_loader)
            history['train_loss'].append(train_loss)
            print(f"Train Loss: {train_loss:.4f}")
            
            # Validate
            if val_loader is not None:
                val_loss = self.validate(val_loader)
                history['val_loss'].append(val_loss)
                print(f"Val Loss: {val_loss:.4f}")
                
                # Learning rate scheduling
                self.scheduler.step(val_loss)
                
                # Early stopping
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                    # Save best model
                    self.save_model(save_path)
                    print(f"✓ Model saved (val_loss: {val_loss:.4f})")
                else:
                    patience_counter += 1
                    if patience_counter >= patience:
                        print(f"\nEarly stopping after {epoch + 1} epochs")
                        break
            else:
                # No validation, save based on train loss
                if train_loss < best_val_loss:
                    best_val_loss = train_loss
                    self.save_model(save_path)
            
            print()
        
        print("Training complete!")
        return history
    
    def save_model(self, path):
        """Save model to disk."""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'embedding_dim': self.model.embedding_dim
        }, path)
    
    @staticmethod
    def load_model(path, model, device='cpu'):
        """
        Load model from disk.
        
        Args:
            path: Path to saved model
            model: SiameseRNNTriplet instance
            device: Device to load model to
        
        Returns:
            model: Loaded model
        """
        checkpoint = torch.load(path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(device)
        return model


class Evaluator:
    """
    Evaluate model performance.
    """
    
    @staticmethod
    def predict_similarity(model, sample1, sample2, device='cpu'):
        """
        Predict similarity between two samples.
        
        Args:
            model: Trained model
            sample1: First sample (seq_len, features)
            sample2: Second sample (seq_len, features)
            device: Device to run on
        
        Returns:
            similarity: Similarity score (0 to 1)
        """
        model.eval()
        
        with torch.no_grad():
            # Convert to tensors
            s1 = torch.FloatTensor(sample1).unsqueeze(0).to(device)
            s2 = torch.FloatTensor(sample2).unsqueeze(0).to(device)
            
            # Get embeddings
            emb1 = model.get_embedding(s1)
            emb2 = model.get_embedding(s2)
            
            # Calculate distance (both embeddings are normalized)
            distance = F.pairwise_distance(emb1, emb2, p=2).item()
            
            # Convert to similarity (distance range is [0, 2] for normalized vectors)
            similarity = 1 - (distance / 2)
            
            return similarity
    
    @staticmethod
    def calculate_metrics(model, test_samples, device='cpu'):
        """
        Calculate authentication metrics (FAR, FRR, EER).
        
        Args:
            model: Trained model
            test_samples: Dict of user_id to samples
            device: Device to run on
        
        Returns:
            metrics: Dict with performance metrics
        """
        genuine_scores = []
        impostor_scores = []
        
        user_ids = list(test_samples.keys())
        
        # Generate genuine scores
        for user_id in user_ids:
            samples = test_samples[user_id]
            if len(samples) < 2:
                continue
            
            for i in range(len(samples)):
                for j in range(i + 1, len(samples)):
                    score = Evaluator.predict_similarity(
                        model, samples[i], samples[j], device
                    )
                    genuine_scores.append(score)
        
        # Generate impostor scores
        for i, user1 in enumerate(user_ids):
            for user2 in user_ids[i+1:]:
                for sample1 in test_samples[user1][:3]:
                    for sample2 in test_samples[user2][:3]:
                        score = Evaluator.predict_similarity(
                            model, sample1, sample2, device
                        )
                        impostor_scores.append(score)
        
        genuine_scores = np.array(genuine_scores)
        impostor_scores = np.array(impostor_scores)
        
        # Calculate EER
        eer, eer_threshold = Evaluator._calculate_eer(
            genuine_scores, impostor_scores
        )
        
        return {
            'genuine_mean': np.mean(genuine_scores),
            'genuine_std': np.std(genuine_scores),
            'impostor_mean': np.mean(impostor_scores),
            'impostor_std': np.std(impostor_scores),
            'eer': eer,
            'eer_threshold': eer_threshold,
            'num_genuine': len(genuine_scores),
            'num_impostor': len(impostor_scores)
        }
    
    @staticmethod
    def _calculate_eer(genuine_scores, impostor_scores):
        """Calculate Equal Error Rate."""
        thresholds = np.linspace(0, 1, 100)
        
        min_diff = float('inf')
        eer = 0
        eer_threshold = 0
        
        for threshold in thresholds:
            far = np.mean(impostor_scores >= threshold)
            frr = np.mean(genuine_scores < threshold)
            
            diff = abs(far - frr)
            if diff < min_diff:
                min_diff = diff
                eer = (far + frr) / 2
                eer_threshold = threshold
        
        return eer, eer_threshold


# Utility functions
def save_checkpoint(model, optimizer, epoch, loss, path):
    """Save training checkpoint."""
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    }, path)


def load_checkpoint(path, model, optimizer=None):
    """Load training checkpoint."""
    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint['model_state_dict'])
    if optimizer:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    return checkpoint['epoch'], checkpoint['loss']


if __name__ == "__main__":
    """
    Demo and testing.
    """
    print("="*70)
    print("Siamese RNN with Triplet Loss - PyTorch Implementation")
    print("="*70)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nUsing device: {device}")
    
    # Create sample data
    sequence_length = 5
    feature_dim = 4
    
    # Simulate 3 users with 10 samples each
    user_samples = {}
    np.random.seed(42)
    for user_id in range(3):
        user_samples[f'user_{user_id}'] = [
            np.random.randn(sequence_length, feature_dim).astype(np.float32) + user_id * 0.5
            for _ in range(10)
        ]
    
    print(f"\nGenerated {len(user_samples)} users with 10 samples each")
    
    # Create model
    model = SiameseRNNTriplet(
        input_dim=feature_dim,
        hidden_dim1=128,
        hidden_dim2=64,
        embedding_dim=32
    )
    
    print(f"\nModel Architecture:")
    print(model)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nTotal parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # Generate triplets
    print("\nGenerating triplets...")
    generator = TripletGenerator()
    anchors, positives, negatives = generator.generate_triplets(
        user_samples, num_triplets=200
    )
    
    print(f"Generated {len(anchors)} triplets")
    print(f"Anchor shape: {anchors.shape}")
    
    # Create dataset and dataloader
    dataset = TripletDataset(anchors, positives, negatives)
    dataloader = DataLoader(dataset, batch_size=16, shuffle=True)
    
    # Create trainer
    trainer = Trainer(model, device=device, learning_rate=0.001)
    
    # Train
    print("\nTraining model...")
    history = trainer.train(
        train_loader=dataloader,
        epochs=20,
        save_path='test_model.pt'
    )
    
    # Test similarity
    print("\n" + "="*70)
    print("Testing Similarity Predictions")
    print("="*70)
    
    sample1 = user_samples['user_0'][0]
    sample2 = user_samples['user_0'][1]  # Same user
    sample3 = user_samples['user_1'][0]  # Different user
    
    sim_same = Evaluator.predict_similarity(model, sample1, sample2, device)
    sim_diff = Evaluator.predict_similarity(model, sample1, sample3, device)
    
    print(f"\nSimilarity (same user): {sim_same:.4f}")
    print(f"Similarity (different user): {sim_diff:.4f}")
    print(f"Difference: {sim_same - sim_diff:.4f}")
    
    if sim_same > sim_diff:
        print("✓ Model correctly distinguishes users!")
    else:
        print("✗ Model needs more training")
    
    # Calculate metrics
    print("\n" + "="*70)
    print("Performance Metrics")
    print("="*70)
    
    metrics = Evaluator.calculate_metrics(model, user_samples, device)
    print(f"\nGenuine scores: {metrics['genuine_mean']:.4f} ± {metrics['genuine_std']:.4f}")
    print(f"Impostor scores: {metrics['impostor_mean']:.4f} ± {metrics['impostor_std']:.4f}")
    print(f"EER: {metrics['eer']*100:.2f}%")
    print(f"EER Threshold: {metrics['eer_threshold']:.4f}")
    
    print("\n" + "="*70)
    print("Demo Complete!")
    print("="*70)