# getting the dataset
words= open('/content/drive/MyDrive/Colab Notebooks/train.txt','r', encoding='utf-8').read().splitlines()
print(len(words))
chars=sorted(list(set(''.join(words))))
# # Add '.' to the list of characters and assign index 0
# chars = ['.'] + chars
#mapping the data from stoi to itos
stoi={s:i for i,s in enumerate(chars)}
stoi['.']=0
itos={i:s for s,i in stoi.items()}

# print(stoi)
print(len(itos))
def build_dataset(words):
  block_size=3 # for the context
  xs,ys=[],[]
  for i in words:
    context=[0]*block_size
    for ch in i+'.':
      ix=stoi[ch]
      xs.append(context)
      ys.append(ix)
      context=context[1:]+[ix] # sliding the window forward
  xs=torch.tensor(xs) # Convert to tensor after the loop
  ys=torch.tensor(ys) # Convert to tensor after the loop
  return xs,ys

n1=int(0.8*len(words))
n2=int(0.9*len(words))
train_words=words[:n1]
val_words=words[n1:n2]
test_words=words[n2:]

import torch

# Use the build_dataset function from the first cell
train_xs, train_ys = build_dataset(train_words)
val_xs, val_ys = build_dataset(val_words)
test_xs, test_ys = build_dataset(test_words)

print(train_xs.shape, train_ys.shape)
print(val_xs.shape, val_ys.shape)
print(test_xs.shape, test_ys.shape)

def get_batch(xs, ys, batch_size=512):
    # Generate random starting indices for the sequences
    ix = torch.randint(0, xs.shape[0] - xs.shape[1], (batch_size,))
    # Extract batches of sequences from xs and the corresponding next character from ys
    batch_xs = torch.stack([xs[i:i+xs.shape[1]] for i in ix])
    batch_ys = torch.stack([ys[i+xs.shape[1]] for i in ix])
    return batch_xs, batch_ys

import matplotlib.pyplot as plt
losses = []

# create the forward pass for training with all the parameters
import torch.nn.functional as F
import torch.nn as nn

# xenc=F.one_hot(xs,num_classes=282).float()
# xenc.shape
embedding = nn.Embedding(num_embeddings=282, embedding_dim=64)
w1=torch.randn((64 * 3,100),requires_grad=True) # Adjusting dimensions for block_size
b1=torch.randn(100,requires_grad=True)
w2=torch.randn((100,282),requires_grad=True)
b2=torch.randn(282,requires_grad=True)
parameters=[embedding.weight,w1,b1,w2,b2]

# training the model with forward and backward pass
batch_size = 32 # Define batch size
for i in range(1000):
  # Get a random batch of data
  ix = torch.randint(0, train_xs.shape[0], (batch_size,))
  batch_xs = train_xs[ix]
  batch_ys = train_ys[ix]

  # print(f"Shape of batch_xs before embedding: {batch_xs.shape}") # Add print statement here
  emb=embedding(batch_xs) # Pass the batch of xs to the embedding layer
  # print(f"Shape of emb before flatten: {emb.shape}") # Keep this print for now
  # Reshape for the linear layer: (batch_size, block_size, embedding_dim) -> (batch_size, block_size * embedding_dim)
  h=torch.tanh(emb.flatten(start_dim=1)@w1+b1)
  logits=h@w2+b2
  loss=F.cross_entropy(logits,batch_ys) # Calculate loss on the batch
  for p in parameters:
    p.grad=None
  loss.backward()
  for p in parameters:
    p.data+= -0.1*p.grad
  losses.append(loss.item())
  # print(loss.item())
print(loss.item())

plt.plot(losses)
plt.xlabel("Step")
plt.ylabel("Loss")
plt.title("Training Loss Curve")
plt.grid(True)
plt.show()

import torch.nn.functional as F

def sample(num_words=10):
    for _ in range(num_words):
        context = [0, 0, 0]  # Start context
        generated = []
        while True:
            x = torch.tensor([context])
            emb = embedding(x)                      # (1, 3, 64)
            h = torch.tanh(emb.view(1, -1) @ w1 + b1)  # (1, 100)
            logits = h @ w2 + b2                     # (1, 282)
            probs = F.softmax(logits, dim=1)         # (1, 282)
            ix = torch.multinomial(probs, num_samples=1).item()  # Sample 1 token
            if ix == stoi['.']:
                break
            generated.append(itos[ix])
            context = context[1:] + [ix]  # Slide context
        print(''.join(generated))

sample(20)  # Generate 20 words
