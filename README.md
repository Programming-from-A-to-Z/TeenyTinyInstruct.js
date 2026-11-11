# TeenyTinyInstruct

This is [TeenyTinyGPT](https://github.com/Programming-from-A-to-Z/TeenyTinyGPT.js) with one more button.

TeenyTinyGPT learns to continue text. That's how every chat model starts out too, and it's called **pre-training**. Then comes **post-training**: more training, on conversations, so the model learns to take turns. This sketch does both. It pre-trains on stories, and when you press the button it keeps training on conversations.

In a test run, this is what the model was writing just before the button was pressed at step 1,000:

```
When the doll. The animal was a very happy and looked back.
```

And 80 steps after:

```
USER:
Tell me a story about a cat.

ASSISTANT:
Once upon a time, in a small house, there lived a little boy named Tim. Tim loved to play outside.
```

It still tells stories. It learned to tell them as a conversation. (It did not learn to listen: the story is about a boy, not a cat. More on that below.)

## Running it

Open `index.html` with a local server (for example the Live Server extension in VS Code, or `npx serve` in this folder). Then:

1. It starts **pre-training** on stories, from random weights. Watch it write: letters, then words, then stories.
2. After 1,000 steps or so, press **pre-training / post-training**. Now it trains on conversations, and what it writes turns into conversations.
3. Type a question in the box at the bottom and press **ask**. Training pauses, and the model's reply appears under the box. The model sees more than that: the sketch writes `USER:`, your question, and `ASSISTANT:`, and the model writes what comes next (the attention boxes show all of it). It stops when it writes `USER` again.
4. Press **pause / train** to keep training. Press the other button again to go back to stories.

Try **ask** before post-training too. The model has never seen a conversation, so it doesn't answer. It keeps writing until its context is full.

**pause / train** stops the training, and the writing with it. The speed slider and the attention boxes (hover over a token) are the same as TeenyTinyGPT's.

## What changed from TeenyTinyGPT

The model (`gpt.js`) and the tokenizer (`bpe.js`) are TeenyTinyGPT's files, unchanged. But now `sketch.js` loads two text data sources and a button switches which one the model trains on.

```js
function switchData() {
  if (data === stories) {
    data = conversations;
  } else {
    data = stories;
  }
}
```

This is the idea of pre-training and post-training. Everything about the model remains the same, only the training text is different.

Some other changes from TeenyTinyGPT.

- The context is 64 tokens instead of 32 since these conversations are longer.
- The tokenizer is first trained on both texts. `USER` and `ASSISTANT` come up so often in the conversations that each one becomes a single token. Real chat models use a format called "ChatML" with special tokens: `<|im_start|>assistant` before a reply and `<|im_end|>` after one.
- I've added an `ask()` button to simulate a chat app. It puts the question into the chat template and the model completes the text.

## The two texts

`data/stories.txt` is 1,371 very short stories from [TinyStories](https://huggingface.co/datasets/roneneldan/TinyStories), a dataset of stories written by GPT-4 using only words a three or four year old would understand. It was made to find out how small a language model can be and still write English.

`data/conversations.txt` is 1,097 conversations, and it's just a text file too. A conversation written out as text, with a label for who is talking, is called a **chat template**:

```
USER:
Tell me a story about a king.

ASSISTANT:
Once upon a time, there was a king. He was a big and strong king who ruled over his kingdom.

```

The `make-conversations.js` script creates examples from the stories. It finds the USER prompt in a story and takes the first two sentences.

```
node make-conversations.js
```

Here, the conversations have nothing new in them except the format. The replies are starts of the stories the model already read.

## What works and what doesn't

The model learns the format instantly!

Also, the replies are almost never copied from `conversations.txt`. A new story starts made up the same way it made up stories before. So pre-training is still core.

It doesn't really follow your prompt. If you ask for a story about a cat, you'll maybe get one about a cat, but most of the time you won't. This model is too small, and the training too simple, to really follow your prompts.

### Real Post-training

- A real model reads trillions of tokens in pre-training and a much smaller amount in post-training. It's knowledge comes almost entirely from pre-training.
- Real post-training also grades (calculates the error for) only the assistant's reply. The model is learning to be an assistant, not a user! Here every token is graded, which keeps the code the same as pre-training. (You can find this setting in Unsloth Studio: "assistant completions only".)

## Things to try

1. Press the button right away, at step 0. Can it learn conversations with no pre-training? What's different about what it writes?
2. Go back and forth. How many steps does it take to forget, and to learn it again?
3. Hover over a token in a reply in the attention boxes. Does it look back at the request?
4. Change the request in `make-conversations.js` ("Write a story about a…"), or the list of things, and run it again.
5. Write your own `data/conversations.txt` by hand, in the same format. What could the USER ask that the stories can answer?

## References

- 📚 [TinyStories: How Small Can Language Models Be and Still Speak Coherent English?](https://arxiv.org/abs/2305.07759) (Eldan and Li, 2023)
- 📚 [LIMA: Less Is More for Alignment](https://arxiv.org/abs/2305.11206) (Zhou et al., 2023): a big model's knowledge comes almost entirely from pre-training, and 1,000 carefully chosen conversations are enough to teach it the format
- 💻 [TRL SFT Trainer](https://huggingface.co/docs/trl/sft_trainer): post-training a real model in Python, including how to train on the assistant's messages only
- 💻 [TeenyTinyGPT](https://github.com/Programming-from-A-to-Z/TeenyTinyGPT.js): how the model itself works

## Data

- `data/stories.txt` is the first 1.1 MB of the validation set of [TinyStories](https://huggingface.co/datasets/roneneldan/TinyStories) (`TinyStoriesV2-GPT4-valid.txt`) by Ronen Eldan and Yuanzhi Li, shared under the [CDLA-Sharing-1.0](https://cdla.dev/sharing-1-0/) license. Curly quotation marks were changed to straight ones.
- `data/conversations.txt` is made from it by `make-conversations.js`.
