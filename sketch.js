// Programming from A to Z
// https://github.com/Programming-from-A-to-Z/A2Z-F26
//
// TeenyTinyInstruct: TeenyTinyGPT, pre-trained on stories and then post-trained on conversations

let storiesFile = 'data/stories.txt'; // pre-training: learn to write stories
let conversationsFile = 'data/conversations.txt'; // post-training: learn to write conversations
let vocabSize = 500; // how many tokens the tokenizer learns
let contextLength = 64; // how many tokens the model can see at once (a whole conversation fits)
let embedSize = 64; // the size of the each token's vector representation (the "embedding")
let batchSize = 32; // how many samples for each training step
let temperature = 0.8;

let tokenizer; // the BPE tokenizer
let stories; // all the stories as a list of token ids
let conversations; // all the conversations as a list of token ids
let data; // what the model is training on right now: the stories or the conversations
let gpt; // the model!

let training = true;
let answering = false; // is the model in the middle of answering a question?
let steps = 0;
let losses = [];

// Everything the model has generated (as token ids), and the training step when each line began
let generated = [];
let lines = [{ step: 0, text: '' }];
let slider;
let question; // the box for typing a question
let answer; // where the model's reply goes

async function setup() {
  createCanvas(1200, 680);
  textFont('monospace');

  // Load the two texts
  let raw = await fetch(storiesFile);
  let storiesText = await raw.text();
  raw = await fetch(conversationsFile);
  let conversationsText = await raw.text();

  // Train the tokenizer on both (this takes a few seconds!)
  // USER and ASSISTANT come up so often that each one gets a token of its own
  tokenizer = new BPE();
  tokenizer.train(storiesText + conversationsText, vocabSize);
  console.log(tokenizer.vocab);
  // Encode both texts!
  stories = tokenizer.encode(storiesText);
  conversations = tokenizer.encode(conversationsText);
  // Pre-training comes first: start with the stories
  data = stories;
  // Start generating from a new line
  generated = tokenizer.encode('\n');

  // Wait for TensorFlow.js to be ready!
  await tf.ready();
  gpt = new TinyGPT(tokenizer.vocab.length, contextLength, embedSize);

  createButton('pause / train').mousePressed(toggleTraining).position(600, 20);
  createButton('pre-training / post-training').mousePressed(switchData).position(700, 20);
  // Speed: how many frames (training steps) per second, from 1 to 60
  createSpan('speed').position(910, 22);
  slider = createSlider(1, 60, 60).position(960, 20);
  // A question for the model, and its reply, below the canvas
  question = createInput('Tell me a story about a dog.').position(20, 690).size(400);
  createButton('ask').mousePressed(ask).position(435, 690);
  answer = createDiv('').position(20, 725).style('font-family', 'monospace');
}

function toggleTraining() {
  training = !training;
  answering = false;
}

// Post-training is the very same training as pre-training, on a different text.
// (Every token is graded here, the USER's lines too. Real post-training
// usually grades only the ASSISTANT's reply.)
function switchData() {
  if (data === stories) {
    data = conversations;
  } else {
    data = stories;
  }
}

// This is what a chat app does: it writes the USER's part of the conversation
// and ASSISTANT: and then the model completes the text.
// (It only works after post-training. Before that, the model has never seen a conversation!)
function ask() {
  // Training pauses while the model answers
  training = false;
  answering = true;
  let prompt = 'USER:\n' + question.value() + '\n\nASSISTANT:\n';
  // The model starts over, with only the prompt to see
  generated = [];
  for (let id of tokenizer.encode(prompt)) {
    // Skip any character the tokenizer has never seen (there's no 7 in the stories!)
    if (id < 0) continue;
    generated.push(id);
  }
  // Clear the last reply
  answer.html('');
}

function draw() {
  background(255);
  if (!gpt) return;

  // The slider sets how many times per second draw() runs
  frameRate(slider.value());

  if (training) {
    let { inputs, targets } = getBatch();
    let loss = gpt.train(inputs, targets);
    tf.dispose([inputs, targets]);
    losses.push(loss);
    steps++;
    // While it's training, the model keeps writing on the canvas
    let next = generateToken();
    addToLines(next);
  } else if (answering) {
    // Paused: the model only writes to answer a question
    let next = generateToken();
    if (next === 'USER') {
      // The model wrote USER: its reply is over, and it's the user's turn!
      answering = false;
    } else {
      // Only the reply is shown, the way a chat app shows it
      answer.html(next, true);
    }
    // It also has to stop when the context is full
    if (generated.length >= contextLength) {
      answering = false;
    }
  }

  drawLosses(20, 40, width - 40, 140);
  drawWriting(20, 200);
  drawAttention(20, 560);
}

// Get a random block of context length tokens, with each target shifted by one token.
// At every position, the right answer is the next token.
function getBatch() {
  let inputs = [];
  let targets = [];
  for (let i = 0; i < batchSize; i++) {
    let start = floor(random(data.length - contextLength - 1));
    inputs.push(data.slice(start, start + contextLength));
    targets.push(data.slice(start + 1, start + contextLength + 1));
  }
  // Turn the inputs and targets into tensors of shape [batch, length] with dtype int32
  return {
    inputs: tf.tensor2d(inputs, [batchSize, contextLength], 'int32'),
    targets: tf.tensor2d(targets, [batchSize, contextLength], 'int32'),
  };
}

function generateToken() {
  // Get the last contextLength tokens we've generated so far
  let context = generated.slice(-contextLength);
  // Ask the model for probabilities for the next token
  let probabilities = gpt.nextToken(context, temperature);
  // Pick a token at random, weighted by the probabilities
  let id = pickIndex(probabilities);
  // Add it to the generated tokens
  generated.push(id);
  return tokenizer.vocab[id];
}

// Add a token to the lines of text on screen
function addToLines(next) {
  let lastLine = lines[lines.length - 1];
  if (next === '\n') {
    // A newline starts a new, empty line
    lines.push({ step: steps, text: '' });
  } else if (lastLine.text.length > 85) {
    // A line that's too long wraps, starting a new line with this token
    lines.push({ step: steps, text: next });
  } else {
    // Otherwise, add the token to the current line
    lastLine.text += next;
  }
}

// Weighted random selection (just like the Markov chain!)
function pickIndex(probabilities) {
  let r = random(1);
  for (let i = 0; i < probabilities.length; i++) {
    r -= probabilities[i];
    if (r <= 0) return i;
  }
  return probabilities.length - 1;
}

// Draw the loss over time
function drawLosses(x, y, w, h) {
  // Guessing at random among every token gives a loss of log(vocabulary size)
  let maxLoss = log(tokenizer.vocab.length);
  noStroke();
  fill(0);
  textSize(14);
  textAlign(LEFT, BOTTOM);
  let status = training ? 'training' : 'paused';
  let phase = data === stories ? 'pre-training on stories' : 'post-training on conversations';
  let latest = nf(losses[losses.length - 1], 1, 2);
  text(`step ${steps}   loss ${latest}   ${status}   ${phase}`, x, y - 10);

  stroke(200);
  noFill();
  rect(x, y, w, h);
  stroke(0, 120, 255);
  beginShape();
  for (let i = 0; i < losses.length; i++) {
    // Spread the steps across the width of the graph
    let px = map(i, 0, max(losses.length - 1, 1), x, x + w);
    // A loss of 0 is at the bottom, maxLoss is at the top
    let py = map(losses[i], 0, maxLoss, y + h, y);
    vertex(px, py);
  }
  endShape();
}

// Draw the most recent lines the model has generated
function drawWriting(x, y) {
  noStroke();
  textAlign(LEFT, TOP);
  textSize(13);
  // Last 20 lines
  let recent = lines.slice(-20);
  for (let i = 0; i < recent.length; i++) {
    fill(150);
    text(`step ${recent[i].step}`, x, y + i * 15);
    fill(0);
    text(recent[i].text, x + 85, y + i * 15);
  }
}

// Draw the model's context window: the last tokens it can see.
// Green shows how much attention one token pays to the tokens before it.
function drawAttention(x, y) {
  let context = generated.slice(-contextLength);
  let weights = gpt.attentionWeights(context);
  let boxSize = 23;
  textSize(14);

  // Tokens have different lengths, so figure out where each box goes first
  let boxes = [];
  let bx = x;
  let by = y;
  for (let i = 0; i < context.length; i++) {
    // Show a new line as \n
    let label = tokenizer.vocab[context[i]].replace('\n', '\\n');
    let w = textWidth(label) + 10;
    // Wrap to the next row
    if (bx + w > width - 20) {
      bx = x;
      by += boxSize + 5;
    }
    boxes.push({ x: bx, y: by, w: w, label: label });
    bx += w + 2;
  }

  // Which token is the mouse over
  let from = context.length - 1;
  for (let i = 0; i < boxes.length; i++) {
    let b = boxes[i];
    if (mouseX > b.x && mouseX < b.x + b.w && mouseY > b.y && mouseY < b.y + boxSize) {
      from = i;
    }
  }

  noStroke();
  fill(0);
  textAlign(LEFT, BOTTOM);
  text('attention:', x, y - 10);

  textAlign(CENTER, CENTER);
  for (let i = 0; i < boxes.length; i++) {
    let b = boxes[i];
    // Get the attention weight from the "from" token to this token
    let weight = weights[from][i];
    // Set alpha according to attention weight
    fill(0, 255, 0, sqrt(weight) * 255);
    stroke(0);
    rect(b.x, b.y, b.w, boxSize);
    noStroke();
    fill(0);
    text(b.label, b.x + b.w / 2, b.y + boxSize / 2);
  }
}
