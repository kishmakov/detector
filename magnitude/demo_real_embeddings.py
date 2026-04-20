"""
Quick demo: magnitude dimension on real roberta-base token embeddings.

Uses a small set of pre-written human texts and AI-style texts (simple,
repetitive patterns) to verify that the pipeline produces the expected
ordering: human > AI in magnitude dimension.

Does not require the original dataset files.

Usage:
    python demo_real_embeddings.py
"""

import sys
import os
import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from src.magnitude import MagnitudeEstimator

# ---------------------------------------------------------------------------
# Sample texts
# ---------------------------------------------------------------------------

# Human-written: varied sentence structure, rich vocabulary, complex ideas
HUMAN_TEXTS = [
    # From example.ipynb (PHD = 10.33)
    "Speaking of festivities, there is one day in China that stands unrivaled - "
    "the first day of the Lunar New Year, commonly referred to as the Spring Festival. "
    "Even if you're generally uninterested in celebratory events, it's hard to resist "
    "the allure of the family reunion dinner, a quintessential aspect of the Spring Festival. "
    "Throughout the meal, family members raise their glasses to toast one another, expressing "
    "wishes for happiness, peace, health, and prosperity in the upcoming year.",

    # Wikipedia-style human text
    "The golden ratio, often denoted by the Greek letter φ (phi), is an irrational number "
    "approximately equal to 1.6180339887. It appears in numerous places in mathematics, art, "
    "architecture, and nature. The ancient Greeks were among the first to study this ratio, "
    "which they found aesthetically pleasing. The Parthenon in Athens, the Great Pyramid of Giza, "
    "and Leonardo da Vinci's Vitruvian Man are often cited as examples where the golden ratio appears, "
    "though scholars debate the extent to which ancient artists consciously employed it.",

    "Photosynthesis is the biochemical process by which chlorophyll-bearing organisms convert "
    "light energy into chemical energy stored in glucose. The process occurs in two stages: "
    "the light-dependent reactions, which capture photons and generate ATP and NADPH, and "
    "the Calvin cycle (light-independent reactions), which use those energy carriers to fix "
    "atmospheric CO2 into organic molecules. The evolution of oxygenic photosynthesis roughly "
    "2.4 billion years ago fundamentally altered Earth's atmosphere and created conditions "
    "that allowed complex multicellular life to emerge.",

    "The Silk Road was not a single road but rather a network of trade routes connecting "
    "East Asia, South Asia, Central Asia, the Middle East, East Africa, and Southern Europe. "
    "For over a millennium, caravans loaded with silk, spices, precious metals, and ideas "
    "traversed thousands of miles of desert, steppe, and mountain terrain. Along the way, "
    "Buddhism, Islam, Christianity, and Zoroastrianism spread beyond their homelands, while "
    "agricultural crops like cotton and paper-making technology migrated westward, permanently "
    "reshaping the civilizations they touched.",

    "Quantum entanglement is a phenomenon in which two or more particles become correlated in "
    "such a way that measuring the state of one particle instantaneously determines the state of "
    "the other, regardless of the distance separating them. Einstein famously dismissed this as "
    "'spooky action at a distance,' arguing it implied either hidden variables or violations of "
    "relativity. Bell's theorem, proved in 1964, provided a mathematical framework for testing "
    "whether nature could be explained by local hidden variables, and subsequent experiments "
    "consistently confirmed quantum mechanics over all classical alternatives.",
]

# AI-style texts: simpler structure, more repetitive, formulaic
AI_TEXTS = [
    "The topic of space exploration is very important. Space exploration helps us learn "
    "about the universe. Scientists study planets and stars in space. Space exploration "
    "has many benefits for humanity. We can learn new things from space exploration. "
    "Space exploration requires advanced technology. Many countries invest in space programs. "
    "Space exploration will continue to be important in the future. The benefits of space "
    "exploration include scientific discoveries and technological advances.",

    "Exercise is good for your health. Regular exercise helps you stay fit and healthy. "
    "There are many types of exercise you can do. Running is a popular form of exercise. "
    "Swimming is also a great exercise. You should exercise at least three times per week. "
    "Exercise helps improve your mental health as well. Many people enjoy exercising outdoors. "
    "Exercise can help you lose weight and build muscle. It is important to stay active and "
    "exercise regularly for a healthy lifestyle.",

    "Technology has changed the world in many ways. Modern technology makes our lives easier. "
    "We use technology every day for communication. Smartphones are an important technology. "
    "The internet is another important technology. Technology helps businesses operate more "
    "efficiently. Technology has also changed the way we learn. Education technology is "
    "very helpful for students. Technology will continue to advance in the coming years. "
    "It is important to use technology responsibly.",

    "Climate change is a serious problem facing our planet. The Earth's temperature is "
    "rising due to greenhouse gases. Carbon dioxide is the main greenhouse gas. "
    "Human activities are causing climate change. We need to reduce our carbon emissions. "
    "Renewable energy can help fight climate change. Solar energy is one type of renewable "
    "energy. Wind energy is another type. We should all try to reduce our carbon footprint. "
    "Climate change is a global problem that requires global solutions.",

    "Reading books is very beneficial. Books help expand your knowledge and vocabulary. "
    "Reading can improve your concentration and focus. There are many different types of books. "
    "Fiction books tell stories and entertain readers. Non-fiction books provide information. "
    "Reading regularly can help improve your writing skills. Many successful people read "
    "every day. Libraries provide access to many books for free. It is important to develop "
    "a habit of reading regularly.",
]


def get_embeddings(text: str, tokenizer, model) -> np.ndarray:
    text = text.replace('\n', ' ').replace('  ', ' ')
    inputs = tokenizer(text, truncation=True, max_length=512, return_tensors='pt')
    with torch.no_grad():
        outp = model(**inputs)
    return outp[0][0].numpy()[1:-1]   # drop CLS, SEP  → (n_tokens, 768)


def main():
    print('Loading roberta-base...')
    tokenizer = AutoTokenizer.from_pretrained('roberta-base')
    model = AutoModel.from_pretrained('roberta-base')
    model.eval()
    print()

    est = MagnitudeEstimator(n_scales=25, max_points=150, n_reruns=5)

    print(f'{"Text":<50}  {"Type":<6}  {"n_tok":>5}  {"mag_dim":>8}')
    print('-' * 75)

    human_dims, ai_dims = [], []

    for i, text in enumerate(HUMAN_TEXTS):
        emb = get_embeddings(text, tokenizer, model)
        d = est.fit_transform(emb)
        label = 'HUMAN'
        print(f'{text[:48]:<50}  {label:<6}  {emb.shape[0]:>5}  {d:>8.4f}')
        if not np.isnan(d):
            human_dims.append(d)

    print()
    for i, text in enumerate(AI_TEXTS):
        emb = get_embeddings(text, tokenizer, model)
        d = est.fit_transform(emb)
        label = 'AI'
        print(f'{text[:48]:<50}  {label:<6}  {emb.shape[0]:>5}  {d:>8.4f}')
        if not np.isnan(d):
            ai_dims.append(d)

    print()
    h_mean = float(np.mean(human_dims))
    a_mean = float(np.mean(ai_dims))
    print(f'Mean magnitude dim — Human: {h_mean:.4f}')
    print(f'Mean magnitude dim — AI:    {a_mean:.4f}')
    print(f'Difference (Human − AI):    {h_mean - a_mean:.4f}')
    print()
    correct_dir = h_mean > a_mean
    print(f'Direction correct (human > AI): {correct_dir}')
    if correct_dir:
        print('=> Magnitude dimension successfully orders human > AI on these samples.')
    else:
        print('=> Direction reversed on these samples (may need more data or tuning).')


if __name__ == '__main__':
    main()
