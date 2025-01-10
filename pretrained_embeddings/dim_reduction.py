#!/gpfs/commons/home/ANON_USER/anaconda3/envs/cuda_env_ne1/bin/python
#SBATCH --job-name=dim_reduction
#SBATCH --mail-type=ALL
#SBATCH --mail-user=ANON_USER@nygenome.org
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=12G
#SBATCH --gres=gpu:1
#SBATCH --time=30:00:00
#SBATCH --output=logs/outputs/autoencoder.txt
#SBATCH --error=logs/errors/autoencoder.txt


ROOT_DIR = 'ROOT_DIR'
WORKING_DIR = f'{ROOT_DIR}/ANON_USER'
import sys
sys.path.append(f'{WORKING_DIR}/pretrained_embeddings')
import torch
import torch.nn as nn
from torch.nn import functional as F
import torch.optim as optim 
import os
from sklearn.model_selection import train_test_split
import pickle
import torch
import time
import numpy as np


# Hyperparameters
latent_dim = 100 
learning_rates = [1e-5, 5e-5,1e-4, 5e-4, 1e-3, 5e-3]
num_epochs = 2000
batch_size = 128
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
rollup_lvl = 5
dataset = '_ct_filter'
patience = 5


path_embed = {
    'bio_clin_bert': f'{WORKING_DIR}/pretrained_embeddings/bio_clin_bert_embeddings_rollup_lvl_{rollup_lvl}{dataset}.pkl',
    'clin_bert': f'{WORKING_DIR}/pretrained_embeddings/clin_bert_embeddings_rollup_lvl_{rollup_lvl}{dataset}.pkl',
    'biogpt': f'{WORKING_DIR}/pretrained_embeddings/biogpt_embeddings_rollup_lvl_{rollup_lvl}{dataset}.pkl',
    'bio_clin_bert_hierarchy': f'{WORKING_DIR}/pretrained_embeddings/bio_clin_bert_embeddings_rollup_lvl_{rollup_lvl}_hierarchy{dataset}.pkl',
    'clin_bert_hierarchy': f'{WORKING_DIR}/pretrained_embeddings/clin_bert_embeddings_rollup_lvl_{rollup_lvl}_hierarchy{dataset}.pkl',
    'biogpt_hierarchy': f'{WORKING_DIR}/pretrained_embeddings/biogpt_embeddings_rollup_lvl_{rollup_lvl}_hierarchy{dataset}.pkl'}




class Autoencoder(nn.Module):
    def __init__(self, input_dim, latent_dim):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, input_dim//2),
            nn.ReLU(),
            nn.Linear(input_dim//2, 256),
            nn.ReLU(),
            nn.Linear(256, latent_dim)  
        )
        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.ReLU(),
            nn.Linear(256, input_dim//2),
            nn.ReLU(),
            nn.Linear(input_dim//2, input_dim) 
        )
    
    def forward(self, x):
        latent = self.encoder(x)
        reconstructed = self.decoder(latent)
        return reconstructed

class EarlyStopping:
    def __init__(self, patience=10, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
        self.best_model = None

    def __call__(self, test_loss, model):
        if self.best_loss is None:
            self.best_loss = test_loss
            self.best_model = model.state_dict().copy()
        elif test_loss > self.best_loss + self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = test_loss
            self.best_model = model.state_dict().copy()
            self.counter = 0
        return self.early_stop

def prepare_embedding_data(model_embeddings):
    model_embeddings_tensor = torch.tensor(model_embeddings, dtype=torch.float32)
    train_data, test_data = train_test_split(model_embeddings_tensor, test_size=0.2, random_state=42)
    return train_data, test_data

def train_autoencoder(embeddings, patience, learning_rate):
    # Prepare data
    train_data, test_data = prepare_embedding_data(embeddings)
    input_dim = train_data.shape[1]
    print(f'Input dim: {input_dim}', flush = True)
    
    # Initialize model
    autoencoder = Autoencoder(input_dim, latent_dim).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(autoencoder.parameters(), lr=learning_rate)
    early_stopping = EarlyStopping(patience=patience)
    
    # Create data loaders
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, shuffle=False)
    
    # Track metrics
    best_test_loss = float('inf')
    train_losses = []
    test_losses = []
    
    # Training loop
    for epoch in range(num_epochs):
        # Training phase
        autoencoder.train()
        running_loss = 0.0
        for batch in train_loader:
            batch = batch.to(device)
            reconstructed = autoencoder(batch)
            loss = criterion(reconstructed, batch)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            
        avg_train_loss = running_loss / len(train_loader)
        train_losses.append(avg_train_loss)
        
        # Testing phase
        autoencoder.eval()
        test_loss = 0.0
        with torch.no_grad():
            for batch in test_loader:
                batch = batch.to(device)
                reconstructed = autoencoder(batch)
                loss = criterion(reconstructed, batch)
                test_loss += loss.item()
                
        avg_test_loss = test_loss / len(test_loader)
        test_losses.append(avg_test_loss)
        
        # Print progress
        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch + 1}/{num_epochs}]', flush = True)
            print(f'Train Loss: {avg_train_loss:.8f}', flush = True)
            print(f'Test Loss: {avg_test_loss:.8f}', flush = True)
        
        # Early stopping check
        if early_stopping(avg_test_loss, autoencoder):
            print(f"Early stopping triggered at epoch {epoch+1}", flush = True)
            print(f"Best test loss: {early_stopping.best_loss:.8f}", flush = True)
            autoencoder.load_state_dict(early_stopping.best_model)
            break
    
    # Return best model and training history
    return {
        'model': autoencoder,
        'best_test_loss': early_stopping.best_loss,
        'final_train_loss': avg_train_loss,
        'train_losses': train_losses,
        'test_losses': test_losses,
        'epochs_trained': epoch + 1
    }

def get_latent_representations(model, embeddings):
    model.eval()
    with torch.no_grad():
        if not isinstance(embeddings, torch.Tensor):
            embeddings = torch.tensor(embeddings, dtype=torch.float32)
        embeddings = embeddings.to(device)
        latent = model.encoder(embeddings)
        return latent.cpu().numpy()

def train_and_save_autoencoder(emb_model_name, patience, learning_rate):
    with open(path_embed[emb_model_name], 'rb') as f:
        embeddings = pickle.load(f)
    emb_model_path = path_embed[emb_model_name][:-4]
    
    results = train_autoencoder(embeddings, patience, learning_rate)
    model_save_path = f'{emb_model_path}_dim_{latent_dim}_lr_{learning_rate}.pth'
    
    # Save the model and training history
    torch.save({
        'model_state_dict': results['model'].state_dict(),
        'best_test_loss': results['best_test_loss'],
        'final_train_loss': results['final_train_loss'],
        'train_losses': results['train_losses'],
        'test_losses': results['test_losses'],
        'epochs_trained': results['epochs_trained']
    }, model_save_path)

    latent_representations =  get_latent_representations(results['model'], embeddings)
    latent_save_path = f'{emb_model_path}_dim_{latent_dim}_lr_{learning_rate}.pkl'
    with open(latent_save_path, 'wb') as f:
        pickle.dump(latent_representations, f)
    
    return {
        'best_test_loss': results['best_test_loss'],
        'epochs_trained': results['epochs_trained'],
        'model_path': model_save_path,
        'latent_path': latent_save_path
    }

def run_all_embeddings():
    """Train autoencoders for all embeddings and track best learning rates"""
    # Dictionary to store results for each embedding
    embedding_results = {}
    
    for model_name in path_embed.keys():
        embedding_results[model_name] = {
            'best_lr': None,
            'best_test_loss': float('inf'),
            'best_results': None
        }
        
        print(f"\nProcessing embedding: {model_name}", flush=True)
        
        # Try each learning rate
        for lr in learning_rates:
            print(f"Testing learning rate: {lr}", flush=True)
            
            try:
                results = train_and_save_autoencoder(model_name, patience, lr)
                
                # Update best results if this learning rate performs better
                if results['best_test_loss'] < embedding_results[model_name]['best_test_loss']:
                    embedding_results[model_name]['best_lr'] = lr
                    embedding_results[model_name]['best_test_loss'] = results['best_test_loss']
                    embedding_results[model_name]['best_results'] = results
                
                print(f"  Current test loss: {results['best_test_loss']:.8f}")
                print(f"  Best test loss so far: {embedding_results[model_name]['best_test_loss']:.8f}")
                print(f"  Best LR so far: {embedding_results[model_name]['best_lr']}")
                
            except Exception as e:
                print(f"Error processing {model_name} with LR {lr}: {str(e)}")
    
    # Save overall results
    results_path = f'{WORKING_DIR}/pretrained_embeddings/autoencoder_best_results.pkl'
    with open(results_path, 'wb') as f:
        pickle.dump(embedding_results, f)
    
    # Print final summary
    print("\nFinal Results Summary:", flush=True)
    print("-" * 50, flush=True)
    for model_name, results in embedding_results.items():
        print(f"\nModel: {model_name}", flush=True)
        print(f"Best Learning Rate: {results['best_lr']}", flush=True)
        print(f"Best Test Loss: {results['best_test_loss']:.8f}", flush=True)
        print(f"Epochs Trained: {results['best_results']['epochs_trained']}", flush=True)
        print(f"Model Path: {os.path.basename(results['best_results']['model_path'])}", flush=True)
        print(f"Embeddings Path: {os.path.basename(results['best_results']['latent_path'])}", flush=True)
    
    return embedding_results

if __name__ == "__main__":
    print("Starting autoencoder training for all embeddings...", flush=True)
    print(f"Target latent dimension: {latent_dim}", flush=True)
    print(f"Device: {device}", flush=True)
    print(f"Learning rates to test: {learning_rates}", flush=True)
    print(f"Early stopping patience: {patience}", flush=True)
    
    try:
        best_results = run_all_embeddings()
        
        # Create a simple results summary file
        summary_path = f'{WORKING_DIR}/pretrained_embeddings/autoencoder_summary.txt'
        with open(summary_path, 'w') as f:
            f.write("Autoencoder Training Results Summary\n")
            f.write(f"Date: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Latent dimension: {latent_dim}\n")
            f.write(f"Learning rates tested: {learning_rates}\n")
            f.write(f"Early stopping patience: {patience}\n\n")
            
            for model_name, results in best_results.items():
                f.write(f"\nModel: {model_name}\n")
                f.write(f"Best Learning Rate: {results['best_lr']}\n")
                f.write(f"Best Test Loss: {results['best_test_loss']:.8f}\n")
                f.write(f"Epochs Trained: {results['best_results']['epochs_trained']}\n")
                f.write(f"Model Path: {results['best_results']['model_path']}\n")
                f.write(f"Embeddings Path: {results['best_results']['latent_path']}\n")
                f.write("-" * 50 + "\n")
        
    except Exception as e:
        print(f"Error in main execution: {str(e)}", flush=True)
        raise