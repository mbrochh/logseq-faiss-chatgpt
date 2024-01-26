# Slides branch

This branch contains the slides for the talk that I gave on 2024-01-26 at
the library@harbourfront.

You can find the slides [here](https://mbrochh.github.io/logseq-faiss-chatgpt/1)

A video of the talk is coming soon.

This is part three of a three part series

1. [Introduction to ChatGPT, Copilot & Whisper](https://mbrochh.github.io/whisper-youtube-transcribe/1)
2. [Summarising books with ChatGPT, Python and Logseq](https://mbrochh.github.io/kindle-scrape-summarise/1)
3. [Talk to your notes with Logseq and ChatGPT](https://mbrochh.github.io/logseq-faiss-chatgpt/1)

## Running the slides

* clone this repo 
* `git checkout slides`
* `cd slidev`
* run `npm install`
* run `npm run dev`
* browse to `http://localhost:3030/`

## Building the slides

* `git checkout slides`
* `cd slidev`
* run `npm run build`
* `rm -rf ../docs`
* run `mv dist ../docs`
* commit and push

## Running the code

* clone this repo
* create a python virtual environment
* `pip install -r requirements.txt`
* Copy `local_settings.py.sample` to `local_settings.py`
* Fill out the values in `local_settings.py`
* Build the database with your vectors: `python -m second_brain.gather_data`
* Ask questions: `python -m second_brain.ask_gpt "What is the meaning of life?"`