// Programming from A to Z
// https://github.com/Programming-from-A-to-Z/A2Z-F26
//
// Turn the stories into conversations and save them as data/conversations.txt
// Run it with: node make-conversations.js
//
// The USER asks for a story about a thing.
// The ASSISTANT replies with the start of a story that has that thing in its first sentence.

import fs from 'fs';

// Things to ask for a story about
let things = ['dog', 'cat', 'bird', 'bear', 'bunny', 'frog', 'fish', 'duck', 'mouse', 'monkey', 'rabbit', 'owl', 'ball', 'toy', 'car', 'truck', 'boat', 'box', 'book', 'hat', 'cake', 'kite', 'tree', 'flower', 'park', 'garden', 'forest', 'house', 'beach', 'king', 'girl', 'boy'];

let text = fs.readFileSync('data/stories.txt', 'utf8');
// A blank line separates one story from the next
let stories = text.split('\n\n');

let conversations = '';
let count = 0;
for (let story of stories) {
  // Split the story into sentences and keep the first two
  let sentences = story.match(/[^.!?]+[.!?]+/g);
  if (!sentences || sentences.length < 2) continue;
  let start = (sentences[0] + sentences[1]).trim();
  // Skip the long ones, and the ones with a line break or someone talking
  if (start.length > 150 || start.includes('\n') || start.includes('"')) continue;

  // One conversation for every thing in the first sentence
  for (let thing of things) {
    let word = new RegExp('\\b' + thing + '\\b');
    if (word.test(sentences[0])) {
      conversations += 'USER:\n';
      conversations += 'Tell me a story about a ' + thing + '.\n\n';
      conversations += 'ASSISTANT:\n';
      conversations += start + '\n\n';
      count++;
    }
  }
}

fs.writeFileSync('data/conversations.txt', conversations);
console.log('saved', count, 'conversations to data/conversations.txt');
