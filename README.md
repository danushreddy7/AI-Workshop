## AI-Workshop(Learn XOR problem sloving using Multilayer Perceptron and PyGAD Program for it)
# Date:18-10-2025
## Reg No:212223040029
## Name: T.DANUSH REDDY

# AIM/OBJECTIES:
To design and implement a program using PyGAD (Genetic Algorithm) to learn the XOR problem and solve it using a Multilayer Perceptron (MLP) approach.
    - To understand the XOR problem, which is not linearly separable and requires a nonlinear model.

    - To learn how Multilayer Perceptrons (MLPs) can model non-linear decision boundaries.

    - To apply PyGAD, a Genetic Algorithm-based optimization library, to train an MLP to predict XOR outputs.

    - To visualize the trained weights and validate the output against the XOR truth table.

# PROGRAM:
```
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# XOR input and output
X = np.array([[0, 0],
              [0, 1],
              [1, 0],
              [1, 1]])
y = np.array([[0], [1], [1], [0]])

# Build the MLP model
model = Sequential()
model.add(Dense(4, input_dim=2, activation='tanh'))   # Hidden layer
model.add(Dense(1, activation='sigmoid'))             # Output layer

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the model (reduced epochs for speed)
model.fit(X, y, epochs=1000, verbose=0)

# Evaluate and test
_, accuracy = model.evaluate(X, y, verbose=0)
print(f"Accuracy: {accuracy * 100:.2f}%")

# Predictions
predictions = model.predict(X, verbose=0)
predictions = np.round(predictions).astype(int)

for i in range(len(X)):
    print(f"Input: {X[i]} => Predicted Output: {predictions[i][0]}, Actual Output: {y[i][0]}")

```
# OUTPUT:

<img width="1102" height="190" alt="image" src="https://github.com/user-attachments/assets/9227d8cb-12ab-4ae4-bf87-cd6ec82c5256" />

# RESULT:
The pyGAD program forThe final model accurancy was successfull
