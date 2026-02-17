"""
Example: Complete pipeline from raw keystrokes to authentication
"""

import torch
import numpy as np
from Capture.feature_extractor import (
    EnrollmentProcessor,
    create_sample_keystroke_data,
    FeatureAnalyzer
)
from siamese import (
    SiameseRNNTriplet,
    Evaluator,
    Trainer
)


class KeystrokeAuthenticator:
    """
    Complete keystroke authentication system.
    Combines feature extraction with neural network.
    """
    
    def __init__(self, model_path=None, device='cpu'):
        """
        Initialize authenticator.
        
        Args:
            model_path: Path to trained model (optional)
            device: 'cpu' or 'cuda'
        """
        self.device = device
        self.model = None
        self.enrollment_processor = EnrollmentProcessor(augment=True)
        self.user_data = {}
        
        if model_path:
            self.load_model(model_path)
    
    def enroll_user(self, 
                   user_id: str,
                   keystroke_samples: list) -> dict:
        """
        Enroll a new user.
        
        Args:
            user_id: Unique user identifier
            keystroke_samples: List of 3 keystroke sequences
        
        Returns:
            result: Enrollment result with status
        """
        try:
            # Process enrollment data
            features_list, normalizer = self.enrollment_processor.process_enrollment(
                keystroke_samples
            )
            
            # Store user data
            self.user_data[user_id] = {
                'features': features_list,
                'normalizer': normalizer,
                'enrolled_at': np.datetime64('now')
            }
            
            return {
                'success': True,
                'user_id': user_id,
                'num_samples': len(features_list),
                'message': 'User enrolled successfully'
            }
            
        except Exception as e:
            return {
                'success': False,
                'user_id': user_id,
                'message': f'Enrollment failed: {str(e)}'
            }
    
    def authenticate(self,
                    user_id: str,
                    keystroke_data: list,
                    threshold: float = 0.75) -> dict:
        """
        Authenticate a user.
        
        Args:
            user_id: User identifier
            keystroke_data: Single keystroke sequence
            threshold: Minimum similarity for authentication
        
        Returns:
            result: Authentication result with metrics
        """
        if user_id not in self.user_data:
            return {
                'authenticated': False,
                'message': 'User not found',
                'user_id': user_id
            }
        
        if self.model is None:
            return {
                'authenticated': False,
                'message': 'Model not loaded',
                'user_id': user_id
            }
        
        try:
            # Get user's normalizer
            user_normalizer = self.user_data[user_id]['normalizer']
            
            # Process authentication sample
            auth_features = self.enrollment_processor.process_authentication(
                keystroke_data,
                user_normalizer
            )
            
            # Calculate similarity with each enrollment sample
            enrolled_features = self.user_data[user_id]['features']
            similarities = []
            
            for enrolled_sample in enrolled_features:
                similarity = Evaluator.predict_similarity(
                    self.model,
                    enrolled_sample,
                    auth_features,
                    self.device
                )
                similarities.append(similarity)
            
            avg_similarity = np.mean(similarities)
            max_similarity = np.max(similarities)
            
            # Authentication decision
            authenticated = avg_similarity >= threshold
            
            return {
                'authenticated': authenticated,
                'user_id': user_id,
                'similarity_score': float(avg_similarity),
                'max_similarity': float(max_similarity),
                'threshold': threshold,
                'message': 'Authentication successful' if authenticated else 'Authentication failed'
            }
            
        except Exception as e:
            return {
                'authenticated': False,
                'message': f'Authentication error: {str(e)}',
                'user_id': user_id
            }
    
    def set_model(self, model):
        """Set the authentication model."""
        self.model = model.to(self.device)
        self.model.eval()
    
    def load_model(self, path):
        """Load model from file."""
        model = SiameseRNNTriplet(input_dim=4, embedding_dim=32)
        model = Trainer.load_model(path, model, self.device)
        self.set_model(model)


def demo_complete_pipeline():
    """
    Demonstrate complete pipeline from raw keystrokes to authentication.
    """
    print("="*70)
    print("Complete Authentication Pipeline Demo")
    print("="*70)
    
    # Create authenticator
    authenticator = KeystrokeAuthenticator(device='cpu')
    
    # Create and train a simple model (in practice, this would be pre-trained)
    print("\nCreating and training model...")
    model = SiameseRNNTriplet(input_dim=4, embedding_dim=32)
    authenticator.set_model(model)
    
    # Enroll users
    print("\n" + "="*70)
    print("ENROLLMENT PHASE")
    print("="*70)
    
    passwords = {
        'alice': 'AlicePass123',
        'bob': 'BobSecure456',
        'charlie': 'Charlie789!'
    }
    
    for user_id, password in passwords.items():
        print(f"\nEnrolling {user_id}...")
        
        # Create 3 enrollment samples
        enrollment_samples = [
            create_sample_keystroke_data(password, base_speed=100 + i*5, variance=15)
            for i in range(3)
        ]
        
        # Enroll
        result = authenticator.enroll_user(user_id, enrollment_samples)
        
        if result['success']:
            print(f"  ✓ {result['message']}")
            print(f"  ✓ Samples: {result['num_samples']}")
        else:
            print(f"  ✗ {result['message']}")
    
    # Authentication attempts
    print("\n" + "="*70)
    print("AUTHENTICATION PHASE")
    print("="*70)
    
    # Genuine attempts
    print("\nGenuine User Attempts:")
    for user_id, password in passwords.items():
        print(f"\n{user_id} attempting to authenticate...")
        
        # Create authentication sample
        auth_sample = create_sample_keystroke_data(password, base_speed=102, variance=18)
        
        # Authenticate
        result = authenticator.authenticate(user_id, auth_sample, threshold=0.5)
        
        if result['authenticated']:
            print(f"  ✓ AUTHENTICATED")
        else:
            print(f"  ✗ REJECTED")
        
        print(f"  Similarity: {result['similarity_score']:.3f}")
        print(f"  Threshold:  {result['threshold']:.3f}")
    
    # Impostor attempt
    print("\n" + "-"*70)
    print("Impostor Attempt:")
    print(f"\nImpostor trying to authenticate as alice...")
    
    # Different typing pattern
    impostor_sample = create_sample_keystroke_data(
        passwords['alice'],
        base_speed=150,  # Much faster
        variance=30       # More variation
    )
    
    result = authenticator.authenticate('alice', impostor_sample, threshold=0.5)
    
    if result['authenticated']:
        print(f"  ✗ IMPOSTOR PASSED (False Accept)")
    else:
        print(f"  ✓ IMPOSTOR REJECTED (Correct)")
    
    print(f"  Similarity: {result['similarity_score']:.3f}")
    print(f"  Threshold:  {result['threshold']:.3f}")
    
    print("\n" + "="*70)
    print("Demo Complete!")
    print("="*70)


if __name__ == "__main__":
    demo_complete_pipeline()