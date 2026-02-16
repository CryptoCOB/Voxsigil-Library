#!/usr/bin/env python3
"""
VantaEcho Convergence Training System
Trains neural network models until convergence rather than fixed epochs
"""
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import json
import asyncio
import websockets
import logging
from datetime import datetime
import matplotlib.pyplot as plt
from collections import deque

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ConvergenceModel(nn.Module):
    """Neural network model for convergence training"""
    def __init__(self, input_size=20, hidden_sizes=[64, 32, 16], output_size=1):
        super(ConvergenceModel, self).__init__()
        
        layers = []
        prev_size = input_size
        
        for hidden_size in hidden_sizes:
            layers.extend([
                nn.Linear(prev_size, hidden_size),
                nn.ReLU(),
                nn.Dropout(0.2)
            ])
            prev_size = hidden_size
        
        layers.append(nn.Linear(prev_size, output_size))
        self.network = nn.Sequential(*layers)
        
    def forward(self, x):
        return self.network(x)

class ConvergenceTracker:
    """Tracks training convergence"""
    def __init__(self, patience=10, min_delta=1e-4, window_size=20):
        self.patience = patience
        self.min_delta = min_delta
        self.window_size = window_size
        self.loss_history = deque(maxlen=window_size)
        self.wait = 0
        self.best_loss = float('inf')
        self.converged = False
        
    def update(self, loss):
        self.loss_history.append(loss)
        
        if loss < self.best_loss - self.min_delta:
            self.best_loss = loss
            self.wait = 0
        else:
            self.wait += 1
            
        # Check for convergence
        if len(self.loss_history) >= self.window_size:
            recent_losses = list(self.loss_history)[-self.window_size//2:]
            if len(recent_losses) > 1:
                loss_variance = np.var(recent_losses)
                if loss_variance < self.min_delta and self.wait >= self.patience:
                    self.converged = True
                    
        return self.converged
    
    def get_status(self):
        return {
            'converged': self.converged,
            'current_loss': self.loss_history[-1] if self.loss_history else float('inf'),
            'best_loss': self.best_loss,
            'patience_remaining': max(0, self.patience - self.wait),
            'loss_variance': np.var(list(self.loss_history)) if len(self.loss_history) > 1 else 0.0
        }

class ConvergenceTrainingCoordinator:
    def __init__(self, host="0.0.0.0", port=8777):
        self.host = host
        self.port = port
        self.model = ConvergenceModel()
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        self.criterion = nn.MSELoss()
        self.convergence_tracker = ConvergenceTracker()
        self.connected_devices = {}
        self.training_active = True
        self.epoch = 0
        self.training_data = self.generate_convergence_dataset()
        
    def generate_convergence_dataset(self, samples=1000):
        """Generate a complex dataset that requires convergence"""
        np.random.seed(42)
        X = np.random.randn(samples, 20)
        
        # Complex function to learn: polynomial + trigonometric + interaction terms
        y = (
            0.5 * np.sum(X[:, :5] ** 2, axis=1) +  # Quadratic terms
            0.3 * np.sin(np.sum(X[:, 5:10], axis=1)) +  # Trigonometric
            0.2 * np.sum(X[:, 10:15] * X[:, 15:20], axis=1) +  # Interaction terms
            0.1 * np.random.randn(samples)  # Noise
        )
        
        return torch.FloatTensor(X), torch.FloatTensor(y).unsqueeze(1)
    
    def train_epoch(self):
        """Train one epoch and return loss"""
        self.model.train()
        X, y = self.training_data
        
        # Batch training
        batch_size = 32
        total_loss = 0
        num_batches = len(X) // batch_size
        
        for i in range(0, len(X), batch_size):
            batch_X = X[i:i+batch_size]
            batch_y = y[i:i+batch_size]
            
            self.optimizer.zero_grad()
            predictions = self.model(batch_X)
            loss = self.criterion(predictions, batch_y)
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
        
        avg_loss = total_loss / num_batches
        return avg_loss
    
    async def handle_device_connection(self, websocket, path):
        """Handle mobile device connections during convergence training"""
        device_id = None
        device_address = websocket.remote_address
        
        try:
            logger.info(f"📱 Device connecting to convergence training from {device_address}")
            
            async for message in websocket:
                try:
                    data = json.loads(message)
                    msg_type = data.get('type')
                    
                    if msg_type == 'register':
                        device_id = data.get('device_id', f"conv_device_{len(self.connected_devices)}")
                        self.connected_devices[device_id] = {
                            'websocket': websocket,
                            'device_info': data,
                            'connected_at': datetime.now().isoformat(),
                            'status': 'training',
                            'epochs_participated': 0
                        }
                        
                        response = {
                            'type': 'convergence_training_started',
                            'device_id': device_id,
                            'training_mode': 'convergence',
                            'model_architecture': '20→64→32→16→1',
                            'dataset_size': len(self.training_data[0]),
                            'convergence_criteria': {
                                'patience': self.convergence_tracker.patience,
                                'min_delta': self.convergence_tracker.min_delta
                            },
                            'timestamp': datetime.now().isoformat()
                        }
                        await websocket.send(json.dumps(response))
                        logger.info(f"✅ Device {device_id} joined convergence training")
                        
                    elif msg_type == 'request_status':
                        convergence_status = self.convergence_tracker.get_status()
                        response = {
                            'type': 'training_status',
                            'epoch': self.epoch,
                            'convergence_status': convergence_status,
                            'connected_devices': len(self.connected_devices),
                            'training_active': self.training_active,
                            'timestamp': datetime.now().isoformat()
                        }
                        await websocket.send(json.dumps(response))
                        
                except json.JSONDecodeError:
                    logger.error(f"Invalid JSON from {device_address}")
                except Exception as e:
                    logger.error(f"Error handling message: {e}")
                    
        except websockets.exceptions.ConnectionClosed:
            logger.info(f"📱 Device {device_id} disconnected")
        except Exception as e:
            logger.error(f"Device connection error: {e}")
        finally:
            if device_id and device_id in self.connected_devices:
                del self.connected_devices[device_id]
                logger.info(f"🚫 Removed device {device_id} from convergence training")
    
    async def broadcast_training_update(self, update_data):
        """Broadcast training updates to all connected devices"""
        if not self.connected_devices:
            return
            
        message = json.dumps(update_data)
        disconnected = []
        
        for device_id, device_info in self.connected_devices.items():
            try:
                await device_info['websocket'].send(message)
                device_info['epochs_participated'] += 1
            except Exception as e:
                logger.error(f"Failed to send update to {device_id}: {e}")
                disconnected.append(device_id)
        
        # Clean up disconnected devices
        for device_id in disconnected:
            del self.connected_devices[device_id]
    
    async def run_convergence_training(self):
        """Run training until convergence"""
        logger.info("🧠 Starting Convergence Training")
        logger.info(f"📊 Dataset: {len(self.training_data[0])} samples, 20 features → 1 output")
        logger.info(f"🎯 Convergence criteria: patience={self.convergence_tracker.patience}, min_delta={self.convergence_tracker.min_delta}")
        
        training_history = []
        
        while self.training_active and not self.convergence_tracker.converged:
            self.epoch += 1
            
            # Train one epoch
            epoch_loss = self.train_epoch()
            
            # Update convergence tracker
            converged = self.convergence_tracker.update(epoch_loss)
            convergence_status = self.convergence_tracker.get_status()
            
            training_history.append({
                'epoch': self.epoch,
                'loss': epoch_loss,
                'converged': converged
            })
            
            # Log progress
            logger.info(f"🚀 Epoch {self.epoch}: Loss={epoch_loss:.6f}, Best={convergence_status['best_loss']:.6f}, "
                       f"Patience={convergence_status['patience_remaining']}, Variance={convergence_status['loss_variance']:.8f}")
            
            # Broadcast update to connected devices
            update = {
                'type': 'epoch_complete',
                'epoch': self.epoch,
                'loss': epoch_loss,
                'convergence_status': convergence_status,
                'model_weights': [p.detach().cpu().numpy().tolist() for p in self.model.parameters()],
                'timestamp': datetime.now().isoformat()
            }
            
            await self.broadcast_training_update(update)
            
            # Check for convergence
            if converged:
                logger.info(f"🎉 CONVERGENCE ACHIEVED at epoch {self.epoch}!")
                logger.info(f"📊 Final loss: {epoch_loss:.6f}")
                logger.info(f"🏆 Best loss: {convergence_status['best_loss']:.6f}")
                
                # Final broadcast
                final_update = {
                    'type': 'training_converged',
                    'final_epoch': self.epoch,
                    'final_loss': epoch_loss,
                    'total_epochs': self.epoch,
                    'convergence_achieved': True,
                    'training_history': training_history[-50:],  # Last 50 epochs
                    'timestamp': datetime.now().isoformat()
                }
                
                await self.broadcast_training_update(final_update)
                break
            
            # Small delay between epochs
            await asyncio.sleep(0.1)
        
        self.training_active = False
        logger.info("🛑 Convergence training completed")
        
        return {
            'converged': self.convergence_tracker.converged,
            'total_epochs': self.epoch,
            'final_loss': epoch_loss,
            'training_history': training_history
        }
    
    async def start_coordinator(self):
        """Start the convergence training coordinator"""
        logger.info("🌌 VantaEcho Convergence Training Coordinator Starting")
        logger.info(f"📡 WebSocket Server: {self.host}:{self.port}")
        logger.info("🧠 Neural Network: 20→64→32→16→1 architecture")
        logger.info("🎯 Training until convergence...")
        
        # Start WebSocket server for device connections
        start_server = websockets.serve(self.handle_device_connection, self.host, self.port)
        
        # Start convergence training
        training_task = asyncio.create_task(self.run_convergence_training())
        
        await asyncio.gather(start_server, training_task)

async def main():
    coordinator = ConvergenceTrainingCoordinator()
    try:
        await coordinator.start_coordinator()
    except KeyboardInterrupt:
        coordinator.training_active = False
        logger.info("🛑 Convergence training stopped by user")
    except Exception as e:
        logger.error(f"Training error: {e}")

if __name__ == "__main__":
    asyncio.run(main())