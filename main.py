import numpy as np
import my_NN as nn
from tensorflow.keras.datasets import mnist
import pygame
import matplotlib.pyplot as plt


# load MNIST datset
(X_train, y_train), (X_test, y_test) = mnist.load_data()

# Preprocess the data
X_train = X_train.reshape(-1, 28 * 28).astype("float32") / 255.0
X_test = X_test.reshape(-1, 28 * 28).astype("float32") / 255.0

# Convert to list of column vectors
X_train = [x.reshape(-1, 1) for x in X_train]
X_test = [x.reshape(-1, 1) for x in X_test]

# convert outputs from digit value to vector form
def vector_convert(digit, num_digits=10):
    vec = np.zeros((num_digits, 1))
    vec[digit] = 1
    return vec

y_train = [vector_convert(y) for y in y_train]
y_test = [vector_convert(y) for y in y_test]


def accuracy(X_batch, y_batch, model):
    correct = 0
    for X, y in zip(X_batch, y_batch):
        output = model.forward(X)
        prediction = np.argmax(output)
        answer = np.argmax(y)
        if prediction == answer:
            correct += 1
    return correct / len(X_batch)



NN = nn.Neural_Network(n_inputs=784, # each pixel
                       n_outputs=10, # number of digits
                       hidden_layers=2, 
                       hidden_size=64,
                       hidden_activation=nn.ReLU(), 
                       output_activation=nn.Softmax(), # digit probabilities
                       learning_rate=0.02)


# pre test
loss = NN.test(X_test, y_test)
acc = accuracy(X_test, y_test, NN)
print(f'Epoch: 0, Loss: {loss:.4f}, Accuracy: {acc:.4f}')

# Train the neural network
batch_size = 32
epochs = 10
for epoch in range(epochs):
    for i in range(0, len(X_train), batch_size):
        X_batch = X_train[i: i + batch_size]
        y_batch = y_train[i: i + batch_size]

        NN.train(X_batch, y_batch)

    # test the neural network after each epoch training
    loss = NN.test(X_test, y_test)
    acc = accuracy(X_test, y_test, NN)
    print(f'Epoch: {epoch+1}, Loss: {loss:.4f}, Accuracy: {acc:.4f}')



def init_bar_chart():
    plt.show(block=False)
    fig, ax = plt.subplots()
    bars = ax.bar(np.arange(10), [0]*10)
    ax.set_ylim(0, 1)
    ax.set_xticks(np.arange(10))
    ax.set_title("Softmax Output")
    return fig, ax, bars

def update_bar_chart(bars, output):
    for bar, val in zip(bars, output):
        bar.set_height(val)
    plt.pause(0.001)



# inititate the pygame window
pygame.init()

WIDTH, HEIGHT = 280, 280
window_color = (0, 0, 0)
window = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Draw A Number")

pixels = np.zeros((28, 28))


# game loop
running = True
clock = pygame.time.Clock()
mouse_down = False

fig, ax, bars = init_bar_chart()
while running:

    window.fill(window_color)

    # draw pixels
    for x in range(28):
        for y in range(28):
            c = 255 * pixels[x, y]
            color = (c, c, c)

            dx = WIDTH / 28
            dy = HEIGHT / 28
            pygame.draw.rect(window, color, (dx * x, dy * y, dx, dy))
    
    # predict the digit
    if True:
        column = pixels.T.reshape(-1, 1)
        output = NN.forward(column)
        digit = np.argmax(output)
        #print(f'Predicted Digit: {digit}')
        update_bar_chart(bars, output.flatten())

    # check for drawing
    if mouse_down:
        x, y = pygame.mouse.get_pos()
        x = round(28 * x / WIDTH - 0.001)
        y = round(28 * y / HEIGHT - 0.001)
        pixels[x, y] = 1
        if (x > 0 and pixels[x-1, y] == 0): pixels[x-1, y] = 0.7
        if (y > 0 and pixels[x, y-1] == 0): pixels[x, y-1] = 0.7
        if (x < 27 and pixels[x+1, y] == 0): pixels[x+1, y] = 0.7
        if (y < 27 and pixels[x, y+1] == 0): pixels[x, y+1] = 0.7


    for event in pygame.event.get():

        if event.type == pygame.QUIT:
            running = False
        
        elif event.type == pygame.MOUSEBUTTONDOWN:
            mouse_down = True
        elif event.type == pygame.MOUSEBUTTONUP:
            mouse_down = False
        
        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_SPACE:
                # clear the pixels
                pixels.fill(0)

        
    pygame.display.update()

    clock.tick(60)

pygame.quit()
