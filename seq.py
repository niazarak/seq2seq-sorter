import numpy as np

def softmax(x: np.array) -> np.array:
    if np.ndim(x) == 2:
        x = x - np.amax(x, axis=1, keepdims=True)
        x = np.exp(x)
        x /= np.sum(x, axis=1, keepdims=True)
    elif np.ndim(x) == 1:
        x = x - np.max(x)
        x = np.exp(x) / np.sum(np.exp(x))
    return x

def loss(x: np.array, y: np.array) -> np.array:
    """
    x - array of one-hot encoded words
    y - labels
    """
    BATCH, SEQ, INPUT = np.shape(x)
    x = np.reshape(x, newshape=(BATCH * SEQ, INPUT))
    y = np.reshape(y, newshape=(BATCH * SEQ))
    ls = np.log(softmax(x)[np.arange(BATCH * SEQ), y])
    ls = -1 * np.sum(ls)
    ls /= BATCH * SEQ
    return ls

def dloss(x: np.array, y: np.array) -> np.array:
    if np.ndim(y) > 2:
        y = np.argmax(y, axis=2)
    BATCH, SEQ, INPUT = np.shape(x)
    x = np.reshape(x, newshape=(BATCH * SEQ, INPUT))
    y = np.reshape(y, newshape=(BATCH * SEQ))
    grad = softmax(x)
    grad[np.arange(BATCH * SEQ), y] -= 1
    return grad


def one_hot(a: np.array, num_classes: int) -> np.array:
    return np.squeeze(np.eye(num_classes)[a.reshape(-1)]).reshape(-1, len(a), num_classes)


def train(X: np.array, y: np.array):
    # X is set of batches of unsorted numbers, y is sorted

    LR = 0.001
    EPOCHS = 30
    HIDDEN_DIM = 200
    INPUTS_DIM = 30
    BATCH_SIZE = X.shape[1]
    SEQ_SIZE = X.shape[2]
    activ = np.tanh

    W_enc_hh = np.random.randn(HIDDEN_DIM, HIDDEN_DIM)
    W_enc_ih = np.random.randn(INPUTS_DIM, HIDDEN_DIM)
    B_enc = np.random.randn(HIDDEN_DIM)
    W_dec_hh = np.random.randn(HIDDEN_DIM, HIDDEN_DIM)
    W_dec_ih = np.random.randn(INPUTS_DIM, HIDDEN_DIM)
    B_dec = np.random.randn(HIDDEN_DIM)
    W_out = np.random.randn(HIDDEN_DIM, INPUTS_DIM)
    B_out = np.random.randn(INPUTS_DIM)

    def enc_forward(emb_input: np.array) -> list:
        B, S, _ = emb_input.shape
        h = np.zeros((B, HIDDEN_DIM))
        enc_cache = []
        for t in range(S):
            emb_inp_t = emb_input[:, t, :]
            h_new = activ(np.dot(h,W_enc_hh) + np.dot(emb_inp_t, W_enc_ih) + B_enc)
            enc_cache.append((emb_inp_t, h, h_new))
            h = h_new
        return enc_cache

    def dec_forward(emb_answer: np.array, h: np.array) -> tuple:
        B, S, _ = emb_answer.shape
        dec_cache = []
        dec_out = np.zeros((B, S, HIDDEN_DIM))
        for t in range(S):
            emb_ans_t = emb_answer[:, t, :]
            h_new = activ(np.dot(h, W_dec_hh) + np.dot(emb_ans_t, W_dec_ih) + B_dec)
            #print('H', h_new)
            dec_out[:, t, :] = h_new
            dec_cache.append((emb_ans_t, h, h_new))
            h = h_new
        return dec_cache, dec_out

    def out_forward(dec_out: np.array) -> np.array:
        B, S, H = dec_out.shape
        dec_out_ = np.reshape(dec_out, newshape=(B * S, -1))
        out = np.dot(dec_out_, W_out) + B_out
        out = np.reshape(out, newshape=(B, S, -1))
        return out

    for epoch in range(EPOCHS):
        print("Epoch:", epoch)
        for inp, ans in zip(X, y):
            # forward pass with teacher forcing
            # embed inputs
            emb_inp = np.zeros((BATCH_SIZE, SEQ_SIZE, INPUTS_DIM))
            emb_ans = np.zeros((BATCH_SIZE, SEQ_SIZE + 1, INPUTS_DIM))
            for i in range(BATCH_SIZE):
                emb_inp[i] = one_hot(inp[i], INPUTS_DIM)
            for i in range(BATCH_SIZE):
                emb_ans[i] = one_hot(ans[i], INPUTS_DIM)

            # encoder
            enc_cache = enc_forward(emb_inp)
            h = enc_cache[-1][-1]

            # decoder
            dec_cache, dec_out = dec_forward(emb_ans, h)
            h = dec_cache[-1][-1]

            # dense
            out = out_forward(dec_out)

            #print("Decoder pred (t/forced):", out)

            # compute loss
            print("Loss:", loss(out, ans))

            dW_enc_hh = np.zeros_like(W_enc_hh)
            dW_enc_ih = np.zeros_like(W_enc_ih)
            dB_enc = np.zeros_like(B_enc)
            dW_dec_hh = np.zeros_like(W_dec_hh)
            dW_dec_ih = np.zeros_like(W_dec_ih)
            dB_dec = np.zeros_like(B_dec)
            dW_out = np.zeros_like(W_out)
            dB_out = np.zeros_like(B_out)

            # backward pass

            # compute dloss
            dout = dloss(out, ans)
            #print("Loss grad:", dout)

            # dense prop
            dec_out_ = np.reshape(dec_out, newshape=(BATCH_SIZE * (SEQ_SIZE +     1), -1))
            dout = np.reshape(dout, newshape=(BATCH_SIZE * (SEQ_SIZE + 1), -1))

            dB_out = np.sum(dout, axis=0)
            dW_out = np.dot(dec_out_.T, dout)

            ddec_out = np.dot(dout, W_out.T)
            ddec_out = np.reshape(ddec_out, newshape=(BATCH_SIZE, (SEQ_SIZE + 1), HIDDEN_DIM))


            dh = 0
            # decoder prop
            for i in reversed(range(SEQ_SIZE + 1)):
                x, h_prev, h_next = dec_cache[i]

                dh_next = ddec_out[:, i, :] + dh

                dt =  dh_next * (1 - h_prev ** 2)
                db = np.sum(dt, axis=0)
                dWhh = np.dot(h_prev.T, dt)
                dh_prev = np.dot(dt, W_dec_hh.T)
                dWih = np.dot(x.T, dt)
                dx = np.dot(dt, W_dec_ih.T)

                dW_dec_hh += dWhh
                dW_dec_ih += dWih
                dB_dec += db

                dh += dh_prev

            # encoder prop
            for i in reversed(range(SEQ_SIZE)):
                x, h_prev, h_next = enc_cache[i]

                dh_next = dh

                dt =  dh_next * (1 - h_prev ** 2)
                db = np.sum(dt, axis=0)
                dWhh = np.dot(h_prev.T, dt)
                dh_prev = np.dot(dt, W_enc_hh.T)
                dWih = np.dot(x.T, dt)
                dx = np.dot(dt, W_enc_ih.T)

                dW_enc_hh += dWhh
                dW_enc_ih += dWih
                dB_enc += db

                dh += dh_prev

            #print(dh)
            W_enc_hh -= LR * dW_enc_hh
            W_enc_ih -= LR * dW_enc_ih
            B_enc -= LR * dB_enc
            W_dec_hh -= LR * dW_dec_hh
            W_dec_ih -= LR * dW_dec_ih
            B_dec -= LR * dB_dec
            W_out -= LR * dW_out
            B_out -= LR * dB_out

    # eval
    print()
    print("Evaluation")
    enc_cache = enc_forward(one_hot(np.array([[2, 2, 1, 3, 7]]), INPUTS_DIM).reshape((1,5,INPUTS_DIM)))
    h = enc_cache[-1][-1]
    answer = []
    word = -1
    for i in range(6):
        dec_cache, dec_out = dec_forward(
                one_hot(np.array([[word]]), INPUTS_DIM), h)
        h = dec_cache[-1][-1]
        out = out_forward(dec_out)
        word = np.argmax(out.flatten()) - 1
        print("Next sorted number", word)
        answer.append(word)

# generate input
TRAIN_SIZE = 100
NUM_COUNT = 5
s = np.arange(NUM_COUNT)
X = np.zeros((TRAIN_SIZE, NUM_COUNT), dtype=np.int32)
y = np.zeros((TRAIN_SIZE, NUM_COUNT + 1), dtype=np.int32)

for i in range(TRAIN_SIZE):
    s = np.random.randint(NUM_COUNT * 2, size=(NUM_COUNT))
    y[i] = np.append([-1] ,np.sort(s))
    X[i] = s

BATCH_SIZE = 10
X = X[:TRAIN_SIZE-(TRAIN_SIZE % BATCH_SIZE)].reshape((TRAIN_SIZE // BATCH_SIZE, BATCH_SIZE, -1))
y = y[:TRAIN_SIZE-(TRAIN_SIZE % BATCH_SIZE)].reshape((TRAIN_SIZE // BATCH_SIZE, BATCH_SIZE, -1))

#print('x:', X)
#print('y:', y)


train(X, y)

