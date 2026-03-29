# Popular tools for detection of Machine-Generated Text

## GPTZero

### Addressed Problems

GPTZero addresses the authorship attribution problem:

- Distinguishing human-written vs AI-generated text
- Detecting partial AI assistance

### Methods

Top level idea:

- Perplexity: how “surprising” the text is to a language model
- Burstiness: variation in sentence structure

More specifically it is based on supervised model (elements of NLP + deep learning) trained on:
- large-scale human text
- large-scale AI-generated text
- adversarially modified text

### Quality Measures

- In-house metrics
- RAID benchmark

### Disadvantages

While good for AI text detection, it has very hight false-positive rate on
human texts: [paper](https://arxiv.org/pdf/2506.23517)

### UI Idea

![GPTZero UI](images/gptzerome_ui.png)

<!-- --------------------------------------------------------------------------- -->

## Originality.AI

### Addressed Problems

Originality.AI addresses trust and authenticity of text content:

- Distinguishing human-written vs AI-generated text
- Detecting partial AI assistance
- Plagiarism + factual reliability

### Methods

In-house supervised model (elements of NLP + deep learning):

### Quality Measures

- In-house metrics
- RAID benchmark


![Originality.ai UI](images/originalityai_ui.png)


