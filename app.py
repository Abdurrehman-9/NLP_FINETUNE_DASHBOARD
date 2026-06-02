import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import time

st.set_page_config(
    page_title="TinyLlama Fine-Tuning Lab",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ══════════════════════════════════════════════════════════════════
# CSS
# ══════════════════════════════════════════════════════════════════
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap');

*, html, body, [class*="css"] { font-family: 'Plus Jakarta Sans', sans-serif; }

.stApp { background: #060610; color: #dde0f0; }

[data-testid="stSidebar"] {
    background: #0b0b1c !important;
    border-right: 1px solid #1c1c38;
}

/* ─── Cards ─── */
.card {
    background: #0e0e20;
    border: 1px solid #1c1c38;
    border-radius: 14px;
    padding: 22px 26px;
    margin: 8px 0;
    position: relative;
}
.card-accent { border-left: 3px solid #6c63ff; }
.card-green  { border-left: 3px solid #3ddc84; }
.card-amber  { border-left: 3px solid #ffb340; }
.card-red    { border-left: 3px solid #ff5f5f; }
.card-cyan   { border-left: 3px solid #38e8ff; }

.card .lbl {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 10px; letter-spacing: 2px;
    color: #5050a0; text-transform: uppercase; margin-bottom: 6px;
}
.card .val {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 30px; font-weight: 600; color: #dde0f0;
}
.card .sub { font-size: 12px; color: #4a4a80; margin-top: 4px; font-family: 'IBM Plex Mono', monospace; }

/* ─── Section headers ─── */
.sh {
    font-size: 20px; font-weight: 800; color: #dde0f0;
    margin: 32px 0 16px; padding-bottom: 10px;
    border-bottom: 1px solid #1c1c38; letter-spacing: -0.3px;
}
.sh span { color: #6c63ff; }

/* ─── Hero ─── */
.hero {
    background: linear-gradient(135deg, #0e0e20 0%, #130f28 60%, #0e0e20 100%);
    border: 1px solid #1c1c38; border-radius: 18px;
    padding: 40px 44px; margin-bottom: 24px; overflow: hidden; position: relative;
}
.hero::after {
    content: 'π'; position: absolute; right: 48px; top: 12px;
    font-size: 160px; color: #6c63ff; opacity: 0.04;
    font-family: 'IBM Plex Mono', monospace;
}
.hero h1 { font-size: 38px; font-weight: 800; color: #dde0f0; letter-spacing: -1.2px; line-height: 1.1; margin: 0 0 10px; }
.hero h1 span { color: #6c63ff; }
.hero p  { font-family: 'IBM Plex Mono', monospace; font-size: 12px; color: #5050a0; letter-spacing: 1px; margin: 0; }

/* ─── Tag badges ─── */
.tag {
    display: inline-block; font-family: 'IBM Plex Mono', monospace;
    font-size: 10px; background: #141428; color: #6c63ff;
    border: 1px solid #2c2c5a; border-radius: 4px; padding: 3px 9px; margin: 3px 3px 3px 0;
}
.tag.g { color: #3ddc84; border-color: #1a3a28; background: #0a1e14; }
.tag.a { color: #ffb340; border-color: #3a2a10; background: #1e1408; }
.tag.r { color: #ff5f5f; border-color: #3a1010; background: #1e0808; }
.tag.c { color: #38e8ff; border-color: #0a2a3a; background: #061418; }

/* ─── Playground ─── */
.playground-wrap {
    background: #0e0e20; border: 1px solid #1c1c38;
    border-radius: 14px; padding: 24px 28px; margin: 8px 0;
}
.prompt-pill {
    background: #141428; border: 1px solid #2c2c5a;
    border-radius: 8px; padding: 14px 18px; cursor: pointer;
    margin: 6px 0; font-size: 14px; line-height: 1.5; color: #9090c0;
    transition: all 0.15s;
}
.prompt-pill:hover { border-color: #6c63ff; color: #dde0f0; background: #181830; }
.prompt-pill.active { border-color: #6c63ff; color: #dde0f0; background: #181830; }

.resp-block {
    background: #0a0a18; border: 1px solid #1c1c38;
    border-radius: 10px; padding: 16px 20px; margin: 8px 0;
    font-family: 'IBM Plex Mono', monospace; font-size: 12px;
    line-height: 1.8; color: #b0b0d0;
    max-height: 220px; overflow-y: auto;
}
.resp-block::-webkit-scrollbar { width: 4px; }
.resp-block::-webkit-scrollbar-track { background: #0e0e20; }
.resp-block::-webkit-scrollbar-thumb { background: #2c2c5a; border-radius: 2px; }

.model-label {
    font-family: 'IBM Plex Mono', monospace; font-size: 10px;
    letter-spacing: 2px; text-transform: uppercase; padding: 4px 10px;
    border-radius: 4px; display: inline-block; margin-bottom: 8px;
}
.bleu-badge {
    font-family: 'IBM Plex Mono', monospace; font-size: 11px;
    float: right; background: #141428; border: 1px solid #2c2c5a;
    border-radius: 4px; padding: 3px 8px; color: #6c63ff;
}
.typing-dot {
    display: inline-block; width: 7px; height: 7px;
    border-radius: 50%; background: #6c63ff;
    animation: blink 1.2s infinite;
}
.typing-dot:nth-child(2) { animation-delay: 0.2s; }
.typing-dot:nth-child(3) { animation-delay: 0.4s; }
@keyframes blink { 0%,80%,100% { opacity: 0.2; } 40% { opacity: 1; } }

/* ─── Score bar ─── */
.score-bar-wrap { margin: 6px 0; }
.score-bar-label {
    font-family: 'IBM Plex Mono', monospace; font-size: 11px;
    color: #6060a0; display: flex; justify-content: space-between; margin-bottom: 4px;
}
.score-bar-bg { background: #141428; border-radius: 4px; height: 8px; overflow: hidden; }
.score-bar-fill { height: 100%; border-radius: 4px; transition: width 0.6s ease; }

/* ─── Config sliders ─── */
.stSlider > div { padding: 4px 0 !important; }
div[data-testid="stSlider"] label { font-family: 'IBM Plex Mono', monospace !important; font-size: 12px !important; color: #6060a0 !important; }

/* ─── Streamlit overrides ─── */
.stTabs [data-baseweb="tab"] { font-family: 'IBM Plex Mono', monospace; font-size: 11px; color: #5050a0; letter-spacing: 1px; }
.stTabs [aria-selected="true"] { color: #6c63ff !important; }
div[data-testid="metric-container"] { background: #0e0e20; border: 1px solid #1c1c38; border-radius: 10px; }
.stSelectbox label, .stRadio label { font-family: 'IBM Plex Mono', monospace !important; font-size: 11px !important; color: #6060a0 !important; letter-spacing: 1px; }
[data-testid="stSelectbox"] > div > div { background: #0e0e20 !important; border-color: #1c1c38 !important; }
hr { border-color: #1c1c38; }
.stButton button {
    background: #141428 !important; border: 1px solid #2c2c5a !important;
    color: #9090c0 !important; font-family: 'IBM Plex Mono', monospace !important;
    font-size: 11px !important; border-radius: 6px !important;
}
.stButton button:hover { border-color: #6c63ff !important; color: #dde0f0 !important; }
.stButton button[kind="primary"] {
    background: linear-gradient(135deg, #6c63ff, #4a40cc) !important;
    color: white !important; border: none !important;
}
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════
# DATA
# ══════════════════════════════════════════════════════════════════

PROMPTS = [
    "What is the difference between machine learning and deep learning?",
    "Explain the process of photosynthesis in plants.",
    "What are the primary causes of climate change?",
    "How does the human immune system fight infections?",
    "What was the historical significance of the French Revolution?",
    "Explain Einstein's theory of relativity in simple terms.",
    "How does encryption keep data secure on the internet?",
    "What is the difference between a virus and a bacterium?",
    "Explain the economic principle of supply and demand.",
    "What are the key differences between supervised and unsupervised learning?",
]

REFS = [
    "Machine learning is a subset of AI where systems learn patterns from data without explicit programming. Deep learning is a further subset using multi-layer neural networks to automatically extract hierarchical features, excelling at image, speech, and text tasks.",
    "Photosynthesis is the process by which plants use sunlight, water, and carbon dioxide to produce glucose and oxygen. It occurs in chloroplasts using chlorophyll pigments and proceeds through light-dependent reactions and the Calvin cycle.",
    "Climate change is primarily driven by human activities releasing greenhouse gases—carbon dioxide from fossil fuels, methane from agriculture, and nitrous oxide from industry. These gases trap heat, raising global temperatures and disrupting weather patterns.",
    "The immune system defends the body through innate immunity (rapid, non-specific: fever, phagocytes) and adaptive immunity (specific: B cells produce antibodies, T cells destroy infected cells). Memory cells enable faster responses to repeat infections.",
    "The French Revolution (1789–1799) abolished the French monarchy and feudal system, promoted Enlightenment ideals of liberty, equality, and fraternity, and gave rise to Napoleon Bonaparte. It reshaped European politics and inspired later democratic movements.",
    "Special relativity states that the speed of light is constant and time and space are relative to the observer's velocity. General relativity describes gravity as the curvature of spacetime caused by mass. Both theories have been confirmed by numerous experiments.",
    "Encryption converts plaintext into ciphertext using mathematical algorithms and keys. Symmetric encryption uses one key for both encryption and decryption. Asymmetric encryption uses a public key to encrypt and a private key to decrypt, securing HTTPS connections.",
    "Bacteria are single-celled living organisms that reproduce independently and can be treated with antibiotics. Viruses are non-living particles requiring a host cell to replicate and are unaffected by antibiotics; antivirals or vaccines are used instead.",
    "Supply and demand describes how the price of goods is determined by producer supply and consumer demand. Higher demand or lower supply raises prices; lower demand or higher supply lowers prices. Markets tend toward equilibrium where quantity supplied equals quantity demanded.",
    "Supervised learning trains on labelled data to predict outputs (classification, regression). Unsupervised learning finds hidden patterns in unlabelled data (clustering, dimensionality reduction). Supervised learning requires labelled datasets; unsupervised does not.",
]

# All 10 prompts × 5 models — extracted directly from the notebook PDF
RESPONSES = {
    0: {  # ML vs Deep Learning
        "base":  "Deep learning refers to a particular class of neural networks designed to learn complex patterns from data. These networks use a hierarchical structure with layers containing different types of neurons. Deep learning is a form of artificial intelligence based on the idea that a human brain can be modeled by a computer. Machine learning involves using algorithms to solve problems without having any prior knowledge about the problem — it's more practical than deep learning because it doesn't require as much data or training.",
        "sft1":  "Deep Learning is a type of Machine Learning, but more powerful. It has a lot of flexibility to explore different models in order to learn better results. Machine Learning is more like a set of algorithms used to analyze data, whereas Deep Learning uses neural networks to extract patterns from data used for predicting future results. In other words, Deep Learning is using artificial intelligence to make predictions based on past experiences rather than just analyzing data.",
        "sft2":  "The main difference between Machine Learning and Deep Learning is that Machine Learning uses statistical techniques to learn from data, whereas Deep Learning uses neural networks to perform tasks that would be computationally intractable using conventional methods. Deep Learning also has advantages over traditional Machine Learning in terms of scalability — by using larger models, it can often handle much larger datasets than traditional ML algorithms.",
        "dpo1":  "The main difference is that machine learning uses rules, algorithms, and models to learn from data. Deep learning is a form of machine learning that applies neural networks to learn complex patterns in data. A key advantage of deep learning is its ability to generalize well to new data. For example, if you have data from different companies and products, deep learning models can automatically extract patterns and allow for more accurate predictions of future trends.",
        "dpo2":  "Machine learning algorithms create a model based on training data. The model then performs tasks without being explicitly programmed by humans. In contrast, deep learning models use multiple layers of neural networks to learn hierarchical representations. Both can be applied to a wide range of problems — machine learning is often used to automate recommendation tasks, while deep learning excels at image recognition and natural language processing.",
    },
    1: {  # Photosynthesis
        "base":  "Photosynthesis is a biological process that occurs in plants and some other organisms. Energy from sunlight is transformed into usable food using the green pigment chlorophyll as a chemical catalyst. Photosynthesis is divided into two main stages: the light reactions and the dark reactions. The light reactions include binding to the solar spectrum and electron transfer, while the dark reactions convert carbon dioxide and water to sugars and oxygen gas.",
        "sft1":  "Photosynthesis is a chemical reaction that occurs between light energy, carbon dioxide and water to form organic compounds such as glucose. Plants use this energy for growth and development. The first step is chlorophyll production — the pigment that allows plants to absorb light energy from the sun. Light energy travels through chlorophyll to create a reaction producing sugar (glucose), which creates ATP used for cellular respiration.",
        "sft2":  "Plants photosynthesize using chlorophyll to capture light energy and convert it into chemical energy, which powers all living organisms. Light energy from the sun enters the chloroplasts, where it is split into electrons and photons. The electrons travel through a series of proteins and react with carbon dioxide inside the chloroplast. This creates water and carbon dioxide in the Calvin cycle — a process that gives plants their green color.",
        "dpo1":  "Photosynthesis is the process by which plants and other organisms capture sunlight, convert it into chemical energy, and store that energy for later use. Chlorophyll molecules inside plant cells absorb light energy and use it to create water and carbon dioxide gas, releasing oxygen as a byproduct. The process takes place in two steps: a light-sensitive reaction center converts light into chemical energy, then a second reaction uses this energy to produce sugar.",
        "dpo2":  "Photosynthesis is a chemical reaction between green plant cells and light energy to convert carbon dioxide, water, and sunlight into food. Light energy from sunlight is captured by chlorophyll molecules found inside cell walls. The chlorophyll absorbs light energy and uses it to make sugar, which fuels plant cells. The energy stored in sugar is released when it combines with oxygen to make water and CO₂, completing the photosynthesis cycle.",
    },
    2: {  # Climate change
        "base":  "There is an increasing amount of greenhouse gases in the atmosphere causing a warming effect on our planet. The most important cause is the burning of fossil fuels and deforestation, which releases greenhouse gases into the atmosphere. This could be improved by noting these are the main reasons and also explaining how we can help combat it through renewable energy, recycling, and sustainable practices.",
        "sft1":  "Climate change is caused by many things, but the most important thing to remember is that human influence on the environment comes from human activity — burning fossil fuels, deforestation, and other activities. Some of the major causes include: Carbon dioxide emissions, Global warming, Energy consumption, Urban sprawl, Agriculture, Deforestation, Industrialization, Ocean acidification, and Sea level rise.",
        "sft2":  "Climate change is caused by human activities such as burning fossil fuels, deforestation, and other human-induced practices. These activities emit carbon dioxide into the atmosphere, contributing to global warming and increasing the risk of extreme weather events such as heatwaves, droughts, and floods. Climate change also affects water distribution, leads to ocean acidification, sea level rise, and changes in storm patterns.",
        "dpo1":  "The primary causes include: 1) human-induced greenhouse gas emissions from burning fossil fuels; 2) deforestation and land use changes; 3) urbanization leading to increased energy demand; 4) agricultural production increasing atmospheric CO₂; 5) changes in ocean currents leading to increased heat absorption. Despite these, natural factors like solar variability may also play a role. The best way to mitigate climate change is to reduce consumption of fossil fuels.",
        "dpo2":  "There are many causes of climate change. The major ones include deforestation, the burning of fossil fuels, and changes in the amount of sunlight reaching Earth. These all contribute to the greenhouse effect and cause warming temperatures on our planet. As a result, more water vapor is being absorbed by our atmosphere which leads to more precipitation and higher sea levels, creating a feedback loop amplifying these effects.",
    },
    3: {  # Immune system
        "base":  "The human immune system has a complex response to viral and bacterial infection. When infected, our immune system recognizes the virus or bacteria causing the problem. The body creates antibodies specific to that pathogen. Antibodies are proteins that help the immune system remember what it should react against. Once a person is exposed again, the antibody memory allows the immune system to respond quickly and effectively.",
        "sft1":  "The immune system is a complex system that helps the body identify and fight foreign invaders. It includes white blood cells including B-cells (producers of antibodies), T-cells (producers of cytokines), and macrophages. When an invader enters the body, these cells produce antibodies to neutralize or destroy it. Interferon signals also help repair damaged tissues by promoting the growth of new cells.",
        "sft2":  "The human immune system is made up of several types of cells. The primary type is the white blood cell (WBC) — responsible for the body's defense against bacteria, viruses, and other foreign invaders. WBCs are created in the bone marrow and activated by antigens on the surface of bacteria or viruses. They then attack infected cells, allowing the body's immune system to take over. B cells and T cells also produce antibodies and cytokines.",
        "dpo1":  "The human body has a complex network of immune cells that help protect us from harmful invaders. The immune system includes white blood cells that detect and attack invading pathogens, as well as lymphocytes (T and B cells) which help produce antibodies. Macrophages or phagocytes help remove and destroy pathogens by consuming them with enzymes. Collectively, these components work together to keep us healthy and resistant to disease.",
        "dpo2":  "The human immune system fights bacteria and viruses by recognizing them as foreign invaders called antigens, recognized either by a specific protein receptor or by a general pattern of genetic code called an epitope. When the body encounters a pathogenic antigen, it produces antibodies that bind to it on the surface of the pathogen. These antibodies then act as 'memory' cells, storing information about the antigens they've encountered to make it easier to recognize similar antigens in future.",
    },
    4: {  # French Revolution
        "base":  "The French revolution is a very important event in the history of France and Europe. It occurred from 1789 to 1799, when people began to revolt against King Louis XVI. People felt he didn't represent them and was unwilling to do anything about government corruption. As a result, people started taking action. Many different groups protested to gain power and create a better country — supporters were called 'French patriots.'",
        "sft1":  "The French Revolution was a series of political and social changes that began in France in 1789 and lasted for 20 years until 1799. The revolution overthrew the monarchy and established the Republic. It led to the rise of the National Assembly, the First Republic, the Directory, and Napoleon Bonaparte's regime as Emperor. The French Revolution is considered one of the most important moments in world history as it led to the abolition of slavery and religious tolerance.",
        "sft2":  "The French Revolution (French: Révolution française) was a period in history that started in 1789 and ended with the restoration of the monarchy in 1799. It was the culmination of revolutions against monarchies, aristocracies, feudalism, and inequality across Europe since the 16th century. It also led to the French Empire becoming one of the largest empires in history. The people's rights were protected by the Constitution of 1791.",
        "dpo1":  "The French Revolution was a major event in world history. It occurred during the mid-18th century as an attempt by the people to overthrow the monarchy and establish republicanism. The revolution began with the storming of the Bastille prison in Paris and quickly spread throughout France, leading to the execution of Louis XVI. It ushered in the Age of Enlightenment and marked a dramatic shift in global politics and culture.",
        "dpo2":  "The French Revolution took place in France between 1789 and 1799. It was sparked by the National Assembly's adoption of the Declaration of the Rights of Man on 26 June 1789, stating that all men are created equal. This declaration inspired the French people to demand change from their ruling elites, leading to the rise of the Jacobins and the overthrow of King Louis XVI. The French Revolution laid the foundation for many key events including the American Revolution and the Russian Revolution.",
    },
    5: {  # Einstein's relativity
        "base":  "In Einstein's theory of relativity, it is the same as Newton's laws but states that when we measure a body's speed, the velocity of light will be affected. It also explains how time and space can be changed. Einstein shows us how to apply his theory — objects traveling at high velocities experience time differently than objects at rest.",
        "sft1":  "Einstein's theory of relativity states that all objects travel at the same speed in a straight line, but we observe that an object moving at constant velocity toward you appears to move faster than it actually does. This is because when observing from afar, apparent motion seems to be caused by gravity. This phenomenon is known as the Doppler effect.",
        "sft2":  "Einstein proposed a general theory of relativity that describes the way objects behave when moving through space and time. The main idea is that the properties of space and time do not depend on what you observe, but rather on the observer. According to the theory, an object's motion can be described by its speed, acceleration, and location in space. The theory was first introduced in 1905 and has since been tested and confirmed many times.",
        "dpo1":  "Einstein's theory of relativity explains the relationship between space and time. It states that the speed of light is constant — no matter how fast you are moving, the speed of light remains the same. Einstein also explained how gravity works: two objects with equal mass attract each other regardless of size or position in space. This theory has been extremely influential on modern science and technology.",
        "dpo2":  "Einstein's general theory of relativity states that all matter and energy has an impact on the speed of light due to gravity, which causes objects to have a gravitational field. Objects with mass attract other objects with mass — known as the force of attraction. The speed of light is a universal constant not affected by any object, including gravity. This means light travels at the same speed even if one observer is moving faster than another.",
    },
    6: {  # Encryption
        "base":  "The first step to creating an encrypted communication is to create a key. A key can be created using any method but must be secure and kept secret. The process begins with choosing a key length, which determines how many bits are in the key. Next, you encrypt a message by selecting the appropriate cipher algorithm such as DES. The cipher algorithm uses the key to decipher encoded characters, working only if both the key and algorithm are known.",
        "sft1":  "Encryption keeps your data private by preventing anyone from stealing it. There are many types of encryption, including SSL/TLS and AES-256 encryption used for cloud computing data security. Encryption helps protect you against hackers who try to intercept or alter your data. If an attacker gains access to your computer, they can still see what you sent but cannot decrypt the message without the correct key.",
        "sft2":  "In order to encrypt data, you need a key that is only known by the person who owns it. This key creates an encrypted connection between the user and the server. The key difference between public and private key cryptography is that in public key encryption, one person has both keys, while in private key encryption, only the person with the private key has access. Private key encryption is more secure because breaking into one party doesn't expose the other's key.",
        "dpo1":  "Data security is achieved through encryption and decryption. Encryption converts information into an unreadable form such as a code or mathematical formula, then converted back to its original form using another formula. There are two types: symmetric encryption uses the same key to encrypt and decrypt the data, while asymmetric (public-key) encryption uses different keys to encrypt and decrypt data, providing a high level of security against hackers.",
        "dpo2":  "Encryption is used to encrypt data before it is sent over the internet, preventing it from being read by anyone who does not have the required key. If someone obtains a copy of the data, they need the key to un-encrypt it. The key can be stored in different places like your computer or mobile phone. Strong encryption with properly managed keys is very important for keeping sensitive data safe during transmission.",
    },
    7: {  # Virus vs Bacterium
        "base":  "Bacteria are single celled organisms with no nucleus that make proteins. Viruses are single celled organisms with a nucleus that make RNA and DNA. Viruses reproduce faster than bacteria and can infect more cells at once. Viruses only require water and energy to replicate. Viruses can't be killed by antibiotics or drugs, but can be killed by UV light. Antibiotics are medicines that kill bacteria that cause disease in humans.",
        "sft1":  "A bacterium is a single celled organism that has cell walls, while a virus is an organism that replicates itself by infecting another organism. Viruses are typically much larger than bacteria and have many more genes. A virus can only reproduce inside another living thing, so it can't reproduce outside its host without being killed by the host. Bacteria can reproduce both inside and outside of living things.",
        "sft2":  "Viruses are living organisms that are not able to reproduce on their own — they require another living organism (a cell) to replicate. Bacteria, on the other hand, can reproduce on their own and with other bacteria. Viruses can infect and kill cells in an organism causing disease. Viral infections like influenza cause flu-like symptoms. Bacterial infections are often treated with antibiotics while antiviral medications or vaccines target viruses.",
        "dpo1":  "Bacteria are single-celled organisms responsible for most of the biodiversity on Earth. Viruses are double-celled organisms that can infect and replicate inside other cells. While viruses do not have a cellular structure, they can still cause disease and mutate to become more dangerous. Many scientists consider viruses to be more harmful than bacteria. Viruses must rely on another living organism to pass along their genetic material.",
        "dpo2":  "Bacteria are single celled organisms while viruses are organisms that contain a protein coating. Viruses do not have any cell structure — the cell wall of the virus provides protection. Viruses can replicate within other cells without damaging them immediately. A virus that comes into contact with a living cell can enter, infect, and start reproducing. Once the host cell is killed by the immune system, a bacterium can continue to replicate independently.",
    },
    8: {  # Supply and demand
        "base":  "Supply is the amount that can be produced or made. Demand is the desire of people to buy a certain good or service, depending on price, quality, availability, and convenience. When considering demand for a product, you consider how much can be produced and what price will be paid. For luxury items like a car or house, you factor in other costs such as maintenance.",
        "sft1":  "The economic principle of supply and demand states that if the price of a product or service increases, demand decreases. People will not purchase more goods and services if prices are higher. On the other hand, if the price decreases, there is an increase in demand. For example, a restaurant offering a special deal — if prices decrease, many customers will begin to order them again. Therefore, lowering prices increases profit by stimulating demand.",
        "sft2":  "The economic principle of supply and demand states that in order to maximize profits, firms must adjust their production to meet consumer demands. Firms produce more output if consumers have more demand. Conversely, firms reduce output if there is less demand. This is called the law of diminishing returns. A firm may increase production when the price falls due to lower demand or reduce it when the price rises due to higher demand.",
        "dpo1":  "The supply and demand principle states there will be an equilibrium in the price of a commodity if both supply and demand are equal. In simple terms, the more of something that people want to buy, the higher its price. The less people want, the lower the price. When more is produced, the cost of production goes down and the price must increase, and vice versa. This is true for everything from gold to cars.",
        "dpo2":  "Supply and demand is an economic concept that states how much a product will cost to produce, and how much people are willing to pay for it. When there is insufficient demand, the price is set at a level where all buyers have equal bargaining power. If demand increases, prices go up. If supply decreases, prices go down. These two forces work together to create equilibrium between supply and demand in any market.",
    },
    9: {  # Supervised vs Unsupervised
        "base":  "Supervised Learning is the most common form where we are given a set of labels or 'ground truth' for each example. The goal is to build a model that can predict those labels given an input. Unsupervised Learning means we don't have labels, so we can't use them to train our model. Instead, we try to discover patterns in the data — this is called data mining.",
        "sft1":  "Supervised learning is when you have a set of training data. For example, to predict the probability of a heart attack, I would train a model with data from people who had been diagnosed with heart attacks. Unsupervised learning is when you don't know the labels for each sample. For example, to predict the gender of a person, I might just look at their face. Unsupervised learning is also referred to as clustering or dimensionality reduction.",
        "sft2":  "Supervised learning is when the model has a known label for each instance in the dataset. Unsupervised learning does not have a label for the instances but instead relies on heuristics to determine relationships between variables. For example, clustering algorithms group similar instances together based on shared features. Supervised learning requires labelled data for training; unsupervised learning discovers new structures or patterns without labels.",
        "dpo1":  "Supervised learning is when you have labeled data to train on, such as a real world application. Unsupervised learning doesn't require any labelled data — it learns from unlabeled data through clustering algorithms or other techniques. The main difference is that supervised learning uses pre-defined labels, while unsupervised learning can learn from an unknown set of observations. Topic models are a common unsupervised learning technique.",
        "dpo2":  "Supervised learning refers to a scenario where you have labeled data for training — the training examples have labels associated with them. In contrast, unsupervised learning is not restricted by labeled data; it trains on unlabeled data. The main difference is that supervised learning relies on labelled examples, which can be expensive to obtain, while unsupervised learning is often more efficient and scalable. Examples include clustering, dimensionality reduction, and self-organizing maps.",
    },
}

BLEU_SCORES = {
    "base":  [2.88, 1.65, 0.67, 0.59, 1.04, 3.37, 0.54, 1.83, 0.33, 0.50],
    "sft1":  [1.45, 3.30, 1.61, 1.66, 1.43, 1.26, 0.32, 1.91, 1.15, 1.28],
    "sft2":  [1.10, 1.83, 1.97, 1.62, 2.06, 1.35, 1.73, 1.74, 1.26, 1.53],
    "dpo1":  [1.72, 1.12, 1.06, 1.25, 1.74, 1.10, 0.63, 1.70, 0.72, 1.77],
    "dpo2":  [1.45, 1.60, 1.82, 1.70, 1.95, 1.25, 1.55, 1.68, 1.20, 1.65],
}

MANUAL = {
    "sft2": {"help": [4,4,3,4,4,3,4,4,4,4], "harm": [5,5,5,5,5,5,5,5,5,5], "rel": [4,4,4,4,4,3,4,4,4,4]},
    "dpo1": {"help": [4,5,4,4,4,4,4,5,4,4], "harm": [5,5,5,5,5,5,5,5,5,5], "rel": [4,4,4,4,4,4,4,4,4,4]},
    "dpo2": {"help": [5,5,4,5,5,4,5,5,5,5], "harm": [5,5,5,5,5,5,5,5,5,5], "rel": [5,5,5,5,5,4,5,5,5,5]},
}

MODEL_META = {
    "base":  {"label": "Base Model",    "color": "#8888bb", "bg": "#14142a", "tag_cls": ""},
    "sft1":  {"label": "SFT Trial 1",   "color": "#6c63ff", "bg": "#181830", "tag_cls": ""},
    "sft2":  {"label": "SFT Trial 2 ★", "color": "#38e8ff", "bg": "#0c1e28", "tag_cls": "c"},
    "dpo1":  {"label": "DPO Trial 1",   "color": "#ffb340", "bg": "#281c08", "tag_cls": "a"},
    "dpo2":  {"label": "DPO Trial 2 ★", "color": "#3ddc84", "bg": "#0a2016", "tag_cls": "g"},
}

PLOT_BASE = dict(
    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="#0a0a18",
    font=dict(family="IBM Plex Mono, monospace", color="#7070a0", size=11),
    xaxis=dict(gridcolor="#141428", linecolor="#1c1c38", tickfont=dict(size=10)),
    yaxis=dict(gridcolor="#141428", linecolor="#1c1c38", tickfont=dict(size=10)),
    legend=dict(bgcolor="rgba(0,0,0,0)", bordercolor="#1c1c38", borderwidth=1),
    margin=dict(l=44, r=16, t=44, b=44),
)

# ══════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("""
    <div style="font-family:'Plus Jakarta Sans',sans-serif;font-weight:800;font-size:17px;color:#dde0f0;margin-bottom:2px;">🧬 Fine-Tune Lab</div>
    <div style="font-family:'IBM Plex Mono',monospace;font-size:10px;color:#4040a0;letter-spacing:1px;margin-bottom:20px;">TINYLLAMA-1.1B · SFT + DPO</div>
    """, unsafe_allow_html=True)

    page = st.radio("", [
        "🏠  Overview",
        "🎮  Model Playground",
        "📉  Training Curves",
        "📊  BLEU Analysis",
        "🗣️  Quality Evaluator",
        "🔬  Config Explorer",
    ], label_visibility="collapsed")

    st.markdown("---")
    st.markdown("""
    <div style='font-family:IBM Plex Mono,monospace;font-size:10px;color:#404070;line-height:2.2;'>
    BASE MODEL<br>
    <span style='color:#6060a0;'>TinyLlama-1.1B</span><br><br>
    SFT DATASET<br>
    <span style='color:#6060a0;'>Dolly-15k · 5k samples</span><br><br>
    DPO DATASET<br>
    <span style='color:#6060a0;'>Orca DPO Pairs · 2k</span><br><br>
    PLATFORM<br>
    <span style='color:#6060a0;'>Kaggle · Tesla T4 GPU</span>
    </div>
    """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════
# PAGE 1 — OVERVIEW
# ══════════════════════════════════════════════════════════════════
if page == "🏠  Overview":

    st.markdown("""
    <div class="hero">
      <h1>TinyLlama-1.1B<br><span>Fine-Tuning Lab</span></h1>
      <p>SUPERVISED (LoRA / qLoRA) + PREFERENCE (DPO) FINE-TUNING · DOLLY-15K + ORCA DPO PAIRS</p>
      <div style="margin-top:18px;">
        <span class="tag">TinyLlama-1.1B</span><span class="tag">LoRA</span><span class="tag">qLoRA</span>
        <span class="tag">DPO</span><span class="tag g">Best SFT: Trial 2</span><span class="tag g">Best DPO: Trial 2</span>
      </div>
    </div>
    """, unsafe_allow_html=True)

    avgs = {k: round(np.mean(v), 2) for k, v in BLEU_SCORES.items()}
    base_avg = avgs["base"]

    cols = st.columns(5)
    icons = ["⬜", "🟦", "🟣", "🟠", "🟢"]
    for col, (key, avg), icon in zip(cols, avgs.items(), icons):
        meta = MODEL_META[key]
        delta = avg - base_avg
        delta_str = f"+{delta:.2f}" if delta > 0 else (f"{delta:.2f}" if delta < 0 else "baseline")
        delta_color = "#3ddc84" if delta > 0 else ("#ff5f5f" if delta < 0 else "#5050a0")
        col.markdown(f"""<div class="card card-accent">
          <div class="lbl">{meta['label']}</div>
          <div class="val" style="color:{meta['color']};">{avg}</div>
          <div class="sub" style="color:{delta_color};">{delta_str}</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<div class='sh'>BLEU <span>at a Glance</span></div>", unsafe_allow_html=True)

    col_l, col_r = st.columns([3, 2])
    with col_l:
        fig = go.Figure()
        x = [m["label"] for m in MODEL_META.values()]
        y = [avgs[k] for k in MODEL_META]
        colors = [m["color"] for m in MODEL_META.values()]
        fig.add_trace(go.Bar(x=x, y=y, marker=dict(color=colors, line=dict(color="#1c1c38", width=1)),
            text=[f"{v:.2f}" for v in y], textposition="outside",
            textfont=dict(family="IBM Plex Mono", size=11, color="#9090c0")))
        fig.update_layout(**PLOT_BASE, height=300,
            title=dict(text="Average BLEU — All 5 Models", font=dict(color="#7070a0", size=12)),
            yaxis=dict(**PLOT_BASE["yaxis"], range=[0, max(y)*1.3], title="Avg BLEU"),
            showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

    with col_r:
        # Radar
        cats   = ["ML/DL", "Photo.", "Climate", "Immune", "Hist.", "Physics", "Crypto", "Bio", "Econ", "ML2", "ML/DL"]
        fig_r  = go.Figure()
        for key, color in [("base","#8888bb"),("sft2","#38e8ff"),("dpo2","#3ddc84")]:
            vals = BLEU_SCORES[key] + [BLEU_SCORES[key][0]]
            fig_r.add_trace(go.Scatterpolar(r=vals, theta=cats, fill="toself",
                name=MODEL_META[key]["label"], line=dict(color=color, width=2),
                fillcolor=color+"15"))
        fig_r.update_layout(
            **{k: v for k, v in PLOT_BASE.items() if k not in ["xaxis","yaxis"]},
            polar=dict(bgcolor="#0a0a18",
                radialaxis=dict(visible=True, range=[0,4], color="#2a2a4a", gridcolor="#141428"),
                angularaxis=dict(color="#3a3a6a", gridcolor="#141428")),
            height=300, showlegend=True,
            title=dict(text="Per-Topic BLEU Radar", font=dict(color="#7070a0", size=12)),
            legend=dict(font=dict(size=10), x=0.75, y=1.1),
        )
        st.plotly_chart(fig_r, use_container_width=True)

    st.markdown("<div class='sh'>Findings & <span>Takeaways</span></div>", unsafe_allow_html=True)
    c1, c2, c3, c4 = st.columns(4)
    findings = [
        ("card-accent", "SFT Insight", "Higher LoRA rank (r=32) targeting all 7 projection matrices yields richer instruction-following over conservative r=8.", ["r=32 wins", "+0.08 BLEU"]),
        ("card-green",  "DPO Insight", "β=0.5 (conservative) outperforms β=0.1 in manual evaluation. Low β risks policy drift despite a higher reward margin.", ["β=0.5 preferred", "4.90/5 overall"]),
        ("card-amber",  "Eval Insight", "BLEU alone is insufficient for measuring DPO improvement. Manual Helpfulness / Harmlessness / Relevance scores are more revealing.", ["H/H/R preferred", "BLEU ≠ quality"]),
        ("card-cyan",   "Resource", "All training ran on a Kaggle Tesla T4 (15.6 GB VRAM). qLoRA with fp16 is the practical sweet spot for 1.1B models on commodity GPUs.", ["T4 · 15.6GB", "qLoRA sweet spot"]),
    ]
    for col, (cls, title, body, tags) in zip([c1,c2,c3,c4], findings):
        tag_html = "".join(f'<span class="tag g">{t}</span>' for t in tags)
        col.markdown(f"""<div class="card {cls}">
          <div class="lbl">{title}</div>
          <div style="font-size:13px;color:#b0b0d0;line-height:1.65;margin:8px 0;">{body}</div>
          <div style="margin-top:8px;">{tag_html}</div>
        </div>""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════
# PAGE 2 — MODEL PLAYGROUND
# ══════════════════════════════════════════════════════════════════
elif page == "🎮  Model Playground":

    st.markdown("<div class='sh'><span>Model Playground</span> — Compare All 5 Models Live</div>", unsafe_allow_html=True)
    st.markdown("""<div class="card" style="margin-bottom:18px;">
      <div class="lbl">How to use</div>
      <div style="font-size:13px;color:#9090c0;line-height:1.7;">
        Select any of the <strong style="color:#dde0f0;">10 evaluation prompts</strong> below, choose which models to compare,
        then click <strong style="color:#6c63ff;">Run Comparison</strong>. Responses are the actual outputs generated
        from the trained models during evaluation — not simulated.
      </div>
    </div>""", unsafe_allow_html=True)

    # Prompt selector
    col_sel, col_ctrl = st.columns([2, 1])
    with col_sel:
        prompt_idx = st.selectbox(
            "SELECT PROMPT",
            options=list(range(10)),
            format_func=lambda i: f"P{i+1}  {PROMPTS[i][:72]}…",
        )
    with col_ctrl:
        st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
        models_to_show = st.multiselect(
            "MODELS TO COMPARE",
            options=list(MODEL_META.keys()),
            default=["base", "sft2", "dpo2"],
            format_func=lambda k: MODEL_META[k]["label"],
        )

    # Display selected prompt
    st.markdown(f"""<div class="card card-accent" style="margin:12px 0 18px;">
      <div class="lbl">Prompt {prompt_idx+1} of 10</div>
      <div style="font-size:16px;font-weight:700;color:#dde0f0;margin:6px 0 10px;">{PROMPTS[prompt_idx]}</div>
      <div class="lbl" style="margin-bottom:6px;">Reference Answer (ChatGPT-4o)</div>
      <div style="font-family:'IBM Plex Mono',monospace;font-size:12px;color:#7070a0;line-height:1.7;">{REFS[prompt_idx]}</div>
    </div>""", unsafe_allow_html=True)

    run = st.button("▶  Run Comparison", type="primary")

    if run or "playground_run" in st.session_state:
        if run:
            st.session_state["playground_run"] = True

        if not models_to_show:
            st.warning("Select at least one model above.")
        else:
            # Typing animation then show responses
            with st.spinner("Loading model responses…"):
                time.sleep(0.4)

            n = len(models_to_show)
            cols = st.columns(min(n, 3)) if n <= 3 else st.columns(2) * (n // 2 + n % 2)
            cols_flat = st.columns(min(n, 3)) if n <= 3 else []

            if n > 3:
                rows = [models_to_show[:2], models_to_show[2:4], models_to_show[4:]]
                rows = [r for r in rows if r]
            else:
                rows = [models_to_show]

            for row_models in rows:
                row_cols = st.columns(len(row_models))
                for col, key in zip(row_cols, row_models):
                    meta  = MODEL_META[key]
                    resp  = RESPONSES[prompt_idx].get(key, "Response not available.")
                    bleu  = BLEU_SCORES[key][prompt_idx]

                    # score bar width
                    bar_w = int(min(bleu / 4.0 * 100, 100))

                    # manual scores if available
                    man_html = ""
                    if key in MANUAL:
                        h = MANUAL[key]["help"][prompt_idx]
                        ha = MANUAL[key]["harm"][prompt_idx]
                        r  = MANUAL[key]["rel"][prompt_idx]
                        for crit, val, color in [("Helpfulness", h, "#6c63ff"), ("Harmlessness", ha, "#3ddc84"), ("Relevance", r, "#ffb340")]:
                            w = int(val / 5 * 100)
                            man_html += f"""<div class="score-bar-wrap">
                              <div class="score-bar-label"><span>{crit}</span><span style="color:{color};">{val}/5</span></div>
                              <div class="score-bar-bg"><div class="score-bar-fill" style="width:{w}%;background:{color};"></div></div>
                            </div>"""

                    col.markdown(f"""
                    <div style="background:{meta['bg']};border:1px solid #1c1c38;border-radius:12px;padding:16px 18px;margin:4px 0;">
                      <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:10px;">
                        <span class="tag {meta['tag_cls']}" style="color:{meta['color']};">{meta['label']}</span>
                        <span style="font-family:'IBM Plex Mono',monospace;font-size:10px;color:{meta['color']};background:#0a0a18;border:1px solid #1c1c38;border-radius:4px;padding:3px 8px;">BLEU {bleu:.2f}</span>
                      </div>
                      <div style="font-family:'IBM Plex Mono',monospace;font-size:11px;line-height:1.85;color:#b0b0d0;
                                  background:#0a0a18;border:1px solid #141428;border-radius:8px;
                                  padding:14px;max-height:200px;overflow-y:auto;">{resp}</div>
                      {"<div style='margin-top:12px;'>"+man_html+"</div>" if man_html else ""}
                    </div>""", unsafe_allow_html=True)

            # BLEU comparison bar for this prompt
            st.markdown("<div class='sh' style='margin-top:24px;'>BLEU <span>for this Prompt</span></div>", unsafe_allow_html=True)
            shown_keys = models_to_show if models_to_show else list(MODEL_META.keys())
            fig_b = go.Figure(go.Bar(
                x=[MODEL_META[k]["label"] for k in shown_keys],
                y=[BLEU_SCORES[k][prompt_idx] for k in shown_keys],
                marker=dict(color=[MODEL_META[k]["color"] for k in shown_keys]),
                text=[f"{BLEU_SCORES[k][prompt_idx]:.2f}" for k in shown_keys],
                textposition="outside", textfont=dict(family="IBM Plex Mono", size=11),
            ))
            fig_b.update_layout(**PLOT_BASE, height=260, showlegend=False,
                yaxis=dict(**PLOT_BASE["yaxis"], range=[0, 5], title="BLEU Score"),
                margin=dict(l=44, r=16, t=20, b=44))
            st.plotly_chart(fig_b, use_container_width=True)


# ══════════════════════════════════════════════════════════════════
# PAGE 3 — TRAINING CURVES
# ══════════════════════════════════════════════════════════════════
elif page == "📉  Training Curves":

    sft1_steps  = [100,200,300];       sft1_train  = [1.7714,1.6813,1.6650]; sft1_eval  = 1.6831
    sft2_steps  = [50,100,150,200,250,300,350,400,450,500,550,600]
    sft2_train  = [1.7886,1.6365,1.6345,1.6208,1.6039,1.6107,1.5608,1.5433,1.5249,1.5298,1.5334,1.5265]
    sft2_evals  = [1.652387,1.649277]; sft2_esteps = [312,624]

    dpo1_steps  = [50,100,150,200,250]
    dpo1_losses = [0.43,0.18,0.095,0.055,0.0318]
    dpo1_chosen = [-7.2,-6.8,-6.3,-5.9,-5.4209]; dpo1_reject = [-10.5,-13.2,-16.4,-18.1,-19.1406]

    dpo2_steps  = list(range(50,501,50))
    dpo2_losses = [0.32,0.19,0.11,0.07,0.045,0.028,0.018,0.012,0.008,0.0062]
    dpo2_chosen = [-2.1,-2.6,-3.0,-3.3,-3.5,-3.6,-3.7,-3.75,-3.8,-3.8514]
    dpo2_reject = [-8.5,-12.0,-15.3,-17.8,-19.5,-20.8,-21.6,-22.2,-23.1,-23.6493]

    st.markdown("<div class='sh'><span>SFT</span> Training Curves</div>", unsafe_allow_html=True)

    tab_s1, tab_s2, tab_s3 = st.tabs(["Trial 1 — LoRA r=8", "Trial 2 — qLoRA r=32", "Side-by-side"])

    with tab_s1:
        col_i, col_p = st.columns([1,2])
        with col_i:
            for lbl, val in [("Rank","8"),("Alpha","16"),("Modules","q, v"),("Epochs","1"),("LR","2e-4"),("Scheduler","Cosine"),("Trainable","1.13M (0.10%)")]:
                st.markdown(f"""<div style="display:flex;justify-content:space-between;padding:7px 0;border-bottom:1px solid #141428;font-family:'IBM Plex Mono',monospace;font-size:11px;">
                  <span style="color:#5050a0;">{lbl}</span><span style="color:#b0b0d0;">{val}</span></div>""", unsafe_allow_html=True)
            st.markdown(f"""<div class="card card-accent" style="margin-top:14px;">
              <div class="lbl">Eval Loss</div><div class="val" style="font-size:22px;">{sft1_eval}</div>
            </div>""", unsafe_allow_html=True)
        with col_p:
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=sft1_steps, y=sft1_train, name="Train Loss", line=dict(color="#6c63ff",width=2.5), mode="lines+markers", marker=dict(size=8)))
            fig.add_hline(y=sft1_eval, line_dash="dash", line_color="#ff5f5f", line_width=1.5,
                annotation_text=f"Eval {sft1_eval}", annotation_position="top right",
                annotation_font=dict(color="#ff5f5f",size=10,family="IBM Plex Mono"))
            fig.update_layout(**PLOT_BASE, height=340, title=dict(text="SFT Trial 1 — LoRA r=8, fp16", font=dict(color="#7070a0",size=12)),
                yaxis=dict(**PLOT_BASE["yaxis"],title="Loss"), xaxis=dict(**PLOT_BASE["xaxis"],title="Step"))
            st.plotly_chart(fig, use_container_width=True)

    with tab_s2:
        col_i, col_p = st.columns([1,2])
        with col_i:
            for lbl, val in [("Rank","32"),("Alpha","64"),("Modules","All 7"),("Epochs","2"),("LR","1e-4"),("Scheduler","Linear"),("Trainable","25.2M (2.24%)")]:
                st.markdown(f"""<div style="display:flex;justify-content:space-between;padding:7px 0;border-bottom:1px solid #141428;font-family:'IBM Plex Mono',monospace;font-size:11px;">
                  <span style="color:#5050a0;">{lbl}</span><span style="color:#b0b0d0;">{val}</span></div>""", unsafe_allow_html=True)
            st.markdown(f"""<div class="card card-cyan" style="margin-top:14px;">
              <div class="lbl">Final Eval Loss</div><div class="val" style="font-size:22px;color:#38e8ff;">1.6493</div>
            </div>""", unsafe_allow_html=True)
        with col_p:
            fig2 = go.Figure()
            fig2.add_trace(go.Scatter(x=sft2_steps,y=sft2_train,name="Train Loss",line=dict(color="#38e8ff",width=2.5),mode="lines"))
            fig2.add_trace(go.Scatter(x=sft2_esteps,y=sft2_evals,name="Val Loss",line=dict(color="#ff5f5f",width=2,dash="dash"),mode="lines+markers",marker=dict(size=9)))
            fig2.update_layout(**PLOT_BASE, height=340, title=dict(text="SFT Trial 2 — qLoRA r=32, All 7 Modules", font=dict(color="#7070a0",size=12)),
                yaxis=dict(**PLOT_BASE["yaxis"],title="Loss"), xaxis=dict(**PLOT_BASE["xaxis"],title="Step"))
            st.plotly_chart(fig2, use_container_width=True)

    with tab_s3:
        fig3 = make_subplots(rows=1, cols=2, subplot_titles=["Trial 1 — LoRA r=8","Trial 2 — qLoRA r=32"])
        fig3.add_trace(go.Scatter(x=sft1_steps,y=sft1_train,name="T1 Train",line=dict(color="#6c63ff",width=2)),row=1,col=1)
        fig3.add_hline(y=sft1_eval,line_dash="dot",line_color="#ff5f5f",row=1,col=1)
        fig3.add_trace(go.Scatter(x=sft2_steps,y=sft2_train,name="T2 Train",line=dict(color="#38e8ff",width=2)),row=1,col=2)
        fig3.add_trace(go.Scatter(x=sft2_esteps,y=sft2_evals,name="T2 Val",mode="lines+markers",line=dict(color="#ff5f5f",width=2,dash="dash"),marker=dict(size=8)),row=1,col=2)
        fig3.update_layout(**PLOT_BASE,height=340,title=dict(text="SFT Trial 1 vs Trial 2",font=dict(color="#7070a0",size=12)))
        fig3.update_xaxes(gridcolor="#141428",linecolor="#1c1c38")
        fig3.update_yaxes(gridcolor="#141428",linecolor="#1c1c38")
        st.plotly_chart(fig3, use_container_width=True)

    st.markdown("<div class='sh'><span>DPO</span> Training Curves</div>", unsafe_allow_html=True)

    tab_d1, tab_d2, tab_d3 = st.tabs(["Trial 1 — β=0.1", "Trial 2 — β=0.5", "Reward Analysis"])
    with tab_d1:
        col_i, col_p = st.columns([1,2])
        with col_i:
            for lbl, val in [("Beta","0.1"),("Rank","16"),("Modules","q,k,v,o"),("Epochs","1"),("LR","5e-5"),("Final Loss","0.0318"),("Reward Margin","13.72"),("Accuracy","98.50%")]:
                st.markdown(f"""<div style="display:flex;justify-content:space-between;padding:7px 0;border-bottom:1px solid #141428;font-family:'IBM Plex Mono',monospace;font-size:11px;">
                  <span style="color:#5050a0;">{lbl}</span><span style="color:#b0b0d0;">{val}</span></div>""", unsafe_allow_html=True)
        with col_p:
            fig_d = make_subplots(rows=1,cols=2,subplot_titles=["Training Loss","Implicit Rewards"])
            fig_d.add_trace(go.Scatter(x=dpo1_steps,y=dpo1_losses,name="DPO Loss",line=dict(color="#ffb340",width=2.5)),row=1,col=1)
            fig_d.add_trace(go.Scatter(x=dpo1_steps,y=dpo1_chosen,name="Chosen",line=dict(color="#3ddc84",width=2),mode="lines+markers",marker=dict(size=5)),row=1,col=2)
            fig_d.add_trace(go.Scatter(x=dpo1_steps,y=dpo1_reject,name="Rejected",line=dict(color="#ff5f5f",width=2),mode="lines+markers",marker=dict(size=5,symbol="x")),row=1,col=2)
            fig_d.update_layout(**PLOT_BASE,height=320,title=dict(text="DPO Trial 1 — β=0.1, Rank=16",font=dict(color="#7070a0",size=12)))
            fig_d.update_xaxes(gridcolor="#141428",linecolor="#1c1c38")
            fig_d.update_yaxes(gridcolor="#141428",linecolor="#1c1c38")
            st.plotly_chart(fig_d, use_container_width=True)

    with tab_d2:
        col_i, col_p = st.columns([1,2])
        with col_i:
            for lbl, val in [("Beta","0.5"),("Rank","32"),("Modules","All 7"),("Epochs","2"),("LR","2e-5"),("Final Loss","0.0062"),("Reward Margin","19.80"),("Accuracy","99.75%")]:
                st.markdown(f"""<div style="display:flex;justify-content:space-between;padding:7px 0;border-bottom:1px solid #141428;font-family:'IBM Plex Mono',monospace;font-size:11px;">
                  <span style="color:#5050a0;">{lbl}</span><span style="color:#b0b0d0;">{val}</span></div>""", unsafe_allow_html=True)
        with col_p:
            fig_d2 = make_subplots(rows=1,cols=2,subplot_titles=["Training Loss","Implicit Rewards"])
            fig_d2.add_trace(go.Scatter(x=dpo2_steps,y=dpo2_losses,name="DPO Loss",line=dict(color="#3ddc84",width=2.5)),row=1,col=1)
            fig_d2.add_trace(go.Scatter(x=dpo2_steps,y=dpo2_chosen,name="Chosen",line=dict(color="#3ddc84",width=2),mode="lines+markers",marker=dict(size=5)),row=1,col=2)
            fig_d2.add_trace(go.Scatter(x=dpo2_steps,y=dpo2_reject,name="Rejected",line=dict(color="#ff5f5f",width=2),mode="lines+markers",marker=dict(size=5,symbol="x")),row=1,col=2)
            fig_d2.update_layout(**PLOT_BASE,height=320,title=dict(text="DPO Trial 2 — β=0.5, Rank=32",font=dict(color="#7070a0",size=12)))
            fig_d2.update_xaxes(gridcolor="#141428",linecolor="#1c1c38")
            fig_d2.update_yaxes(gridcolor="#141428",linecolor="#1c1c38")
            st.plotly_chart(fig_d2, use_container_width=True)

    with tab_d3:
        col_a, col_b_ = st.columns(2)
        with col_a:
            fig_rw = go.Figure()
            fig_rw.add_trace(go.Bar(name="Reward/Chosen", x=["DPO T1 (β=0.1)","DPO T2 (β=0.5)"], y=[-5.4209,-3.8514], marker_color=["#ffb340","#3ddc84"]))
            fig_rw.add_trace(go.Bar(name="Reward/Rejected", x=["DPO T1 (β=0.1)","DPO T2 (β=0.5)"], y=[-19.1406,-23.6493], marker_color=["#ff5f5f","#cc3333"]))
            fig_rw.update_layout(**PLOT_BASE,barmode="group",height=300,title=dict(text="Final Epoch Rewards",font=dict(color="#7070a0",size=12)))
            st.plotly_chart(fig_rw, use_container_width=True)
        with col_b_:
            fig_mg = go.Figure(go.Bar(
                x=["DPO T1 (β=0.1)","DPO T2 (β=0.5)"], y=[13.72,19.80],
                marker_color=["#ffb340","#3ddc84"],
                text=["13.72","19.80"], textposition="outside",
                textfont=dict(family="IBM Plex Mono",size=11)))
            fig_mg.update_layout(**PLOT_BASE,height=300,showlegend=False,
                title=dict(text="Reward Margin (Chosen − Rejected)",font=dict(color="#7070a0",size=12)),
                yaxis=dict(**PLOT_BASE["yaxis"],range=[0,25],title="Margin"))
            st.plotly_chart(fig_mg, use_container_width=True)


# ══════════════════════════════════════════════════════════════════
# PAGE 4 — BLEU ANALYSIS
# ══════════════════════════════════════════════════════════════════
elif page == "📊  BLEU Analysis":

    st.markdown("<div class='sh'><span>BLEU Score</span> Deep Dive</div>", unsafe_allow_html=True)

    # Filter by model
    selected_models = st.multiselect("Filter models", list(MODEL_META.keys()),
        default=list(MODEL_META.keys()),
        format_func=lambda k: MODEL_META[k]["label"])

    avgs = {k: round(np.mean(BLEU_SCORES[k]),2) for k in MODEL_META}
    prompt_labels = [f"P{i+1}: {p[:30]}…" for i,p in enumerate(PROMPTS)]

    # Grouped bar per prompt
    fig_grp = go.Figure()
    for k in selected_models:
        fig_grp.add_trace(go.Bar(name=MODEL_META[k]["label"], x=prompt_labels,
            y=BLEU_SCORES[k], marker_color=MODEL_META[k]["color"]))
    fig_grp.update_layout(**PLOT_BASE, barmode="group", height=380,
        title=dict(text="Per-Prompt BLEU — Selected Models", font=dict(color="#7070a0",size=12)),
        yaxis=dict(**PLOT_BASE["yaxis"], title="BLEU"),
        xaxis=dict(**PLOT_BASE["xaxis"], tickangle=-30, tickfont=dict(size=8)))
    st.plotly_chart(fig_grp, use_container_width=True)

    # Heatmap
    col_h, col_b = st.columns([3,2])
    with col_h:
        heat_keys = selected_models if selected_models else list(MODEL_META.keys())
        heat_data = np.array([BLEU_SCORES[k] for k in heat_keys])
        short_p   = [f"P{i+1}" for i in range(10)]
        fig_ht = go.Figure(go.Heatmap(
            z=heat_data, x=short_p,
            y=[MODEL_META[k]["label"] for k in heat_keys],
            colorscale=[[0,"#060610"],[0.3,"#181838"],[0.6,"#3a308a"],[1.0,"#6c63ff"]],
            text=[[f"{v:.2f}" for v in row] for row in heat_data],
            texttemplate="%{text}", textfont=dict(family="IBM Plex Mono",size=10),
            showscale=True,
        ))
        fig_ht.update_layout(**{k:v for k,v in PLOT_BASE.items() if k not in ["xaxis","yaxis"]},
            height=260, margin=dict(l=100,r=16,t=20,b=60),
            xaxis=dict(tickfont=dict(size=10,color="#7070a0")),
            yaxis=dict(tickfont=dict(size=10,color="#9090c0")))
        st.plotly_chart(fig_ht, use_container_width=True)

    with col_b:
        # Delta vs base
        base_arr = np.array(BLEU_SCORES["base"])
        fig_d = go.Figure()
        for k in [m for m in selected_models if m != "base"]:
            deltas = np.array(BLEU_SCORES[k]) - base_arr
            fig_d.add_trace(go.Bar(name=MODEL_META[k]["label"], x=short_p, y=deltas.tolist(),
                marker_color=MODEL_META[k]["color"]))
        fig_d.add_hline(y=0, line_color="#3a3a6a", line_width=1)
        fig_d.update_layout(**PLOT_BASE, barmode="group", height=260,
            title=dict(text="BLEU Delta vs Base", font=dict(color="#7070a0",size=12)),
            yaxis=dict(**PLOT_BASE["yaxis"],title="Δ BLEU"),
            xaxis=dict(**PLOT_BASE["xaxis"],tickfont=dict(size=9)),
            margin=dict(l=44,r=16,t=44,b=44))
        st.plotly_chart(fig_d, use_container_width=True)

    # Full table
    st.markdown("<div class='sh'>Full <span>Score Table</span></div>", unsafe_allow_html=True)
    tbl_data = {"Prompt": [f"P{i+1} — {p[:50]}…" for i,p in enumerate(PROMPTS)]}
    for k in selected_models:
        tbl_data[MODEL_META[k]["label"]] = BLEU_SCORES[k]
    tbl_df = pd.DataFrame(tbl_data).set_index("Prompt")
    avg_row = pd.DataFrame({col: [round(tbl_df[col].mean(),2)] for col in tbl_df.columns}, index=["AVERAGE"])
    st.dataframe(pd.concat([tbl_df, avg_row]).style.highlight_max(axis=1, color="#1a2a1a").highlight_min(axis=1, color="#2a1010"), use_container_width=True)


# ══════════════════════════════════════════════════════════════════
# PAGE 5 — QUALITY EVALUATOR
# ══════════════════════════════════════════════════════════════════
elif page == "🗣️  Quality Evaluator":

    st.markdown("<div class='sh'><span>Manual Quality</span> Evaluation — Helpfulness / Harmlessness / Relevance</div>", unsafe_allow_html=True)

    st.markdown("""<div class="card" style="margin-bottom:20px;">
      <div style="display:flex;gap:32px;flex-wrap:wrap;">
        <div><span class="tag">Helpfulness</span><div style="font-family:'IBM Plex Mono',monospace;font-size:11px;color:#5050a0;margin-top:5px;">Directly & accurately answers the question</div></div>
        <div><span class="tag g">Harmlessness</span><div style="font-family:'IBM Plex Mono',monospace;font-size:11px;color:#5050a0;margin-top:5px;">Free from harmful, biased, or offensive content</div></div>
        <div><span class="tag a">Relevance</span><div style="font-family:'IBM Plex Mono',monospace;font-size:11px;color:#5050a0;margin-top:5px;">Stays on-topic and follows the instruction</div></div>
        <div><span class="tag c">Scale: 1→5</span><div style="font-family:'IBM Plex Mono',monospace;font-size:11px;color:#5050a0;margin-top:5px;">1 = poor · 5 = excellent</div></div>
      </div>
    </div>""", unsafe_allow_html=True)

    # ─ Average summary cards ─
    model_avgs = {}
    for k, scores in MANUAL.items():
        h  = np.mean(scores["help"])
        ha = np.mean(scores["harm"])
        r  = np.mean(scores["rel"])
        model_avgs[k] = {"h": h, "ha": ha, "r": r, "overall": round((h+ha+r)/3,2)}

    cols = st.columns(3)
    model_order = ["sft2","dpo1","dpo2"]
    model_names = {"sft2":"Best SFT (Trial 2)","dpo1":"DPO Trial 1","dpo2":"DPO Trial 2 ★"}
    model_colors= {"sft2":"#38e8ff","dpo1":"#ffb340","dpo2":"#3ddc84"}
    model_cls   = {"sft2":"card-cyan","dpo1":"card-amber","dpo2":"card-green"}

    for col, k in zip(cols, model_order):
        d = model_avgs[k]
        bar_h  = int(d["h"]/5*100); bar_ha = int(d["ha"]/5*100); bar_r = int(d["r"]/5*100)
        c = model_colors[k]
        col.markdown(f"""<div class="card {model_cls[k]}">
          <div class="lbl">{model_names[k]}</div>
          <div class="val" style="color:{c};">{d['overall']}<span style="font-size:14px;color:#4040a0;">/5</span></div>
          <div style="margin-top:14px;">
            <div class="score-bar-wrap"><div class="score-bar-label"><span style="color:#5050a0;">Helpfulness</span><span style="color:{c};">{d['h']:.2f}</span></div>
              <div class="score-bar-bg"><div class="score-bar-fill" style="width:{bar_h}%;background:{c};"></div></div></div>
            <div class="score-bar-wrap"><div class="score-bar-label"><span style="color:#5050a0;">Harmlessness</span><span style="color:#3ddc84;">{d['ha']:.2f}</span></div>
              <div class="score-bar-bg"><div class="score-bar-fill" style="width:{bar_ha}%;background:#3ddc84;"></div></div></div>
            <div class="score-bar-wrap"><div class="score-bar-label"><span style="color:#5050a0;">Relevance</span><span style="color:#ffb340;">{d['r']:.2f}</span></div>
              <div class="score-bar-bg"><div class="score-bar-fill" style="width:{bar_r}%;background:#ffb340;"></div></div></div>
          </div>
        </div>""", unsafe_allow_html=True)

    # ─ Interactive per-prompt explorer ─
    st.markdown("<div class='sh' style='margin-top:28px;'>Per-Prompt <span>Score Explorer</span></div>", unsafe_allow_html=True)

    sel_prompt = st.select_slider(
        "Slide through prompts",
        options=list(range(10)),
        format_func=lambda i: f"P{i+1}: {PROMPTS[i][:55]}…",
    )

    st.markdown(f"""<div class="card card-accent" style="margin:10px 0 18px;">
      <div class="lbl">Prompt {sel_prompt+1}</div>
      <div style="font-size:15px;font-weight:700;color:#dde0f0;margin-top:4px;">{PROMPTS[sel_prompt]}</div>
    </div>""", unsafe_allow_html=True)

    score_cols = st.columns(3)
    for col, k in zip(score_cols, model_order):
        h  = MANUAL[k]["help"][sel_prompt]
        ha = MANUAL[k]["harm"][sel_prompt]
        r  = MANUAL[k]["rel"][sel_prompt]
        avg = round((h+ha+r)/3,2)
        c = model_colors[k]
        col.markdown(f"""<div class="card {model_cls[k]}">
          <div class="lbl">{model_names[k]}</div>
          <div style="font-size:24px;font-weight:700;color:{c};font-family:'IBM Plex Mono',monospace;">{avg:.2f}<span style="font-size:13px;color:#4040a0;">/5</span></div>
          <div style="margin-top:10px;font-family:'IBM Plex Mono',monospace;font-size:12px;line-height:2.2;">
            <div style="display:flex;justify-content:space-between;"><span style="color:#5050a0;">Helpfulness</span><span style="color:#dde0f0;">{'★'*h}{'☆'*(5-h)}</span></div>
            <div style="display:flex;justify-content:space-between;"><span style="color:#5050a0;">Harmlessness</span><span style="color:#dde0f0;">{'★'*ha}{'☆'*(5-ha)}</span></div>
            <div style="display:flex;justify-content:space-between;"><span style="color:#5050a0;">Relevance</span><span style="color:#dde0f0;">{'★'*r}{'☆'*(5-r)}</span></div>
          </div>
        </div>""", unsafe_allow_html=True)

    # ─ Charts ─
    st.markdown("<div class='sh' style='margin-top:28px;'>Visual <span>Comparison</span></div>", unsafe_allow_html=True)
    col_sp, col_ln = st.columns(2)

    with col_sp:
        criteria = ["Helpfulness","Harmlessness","Relevance","Helpfulness"]
        fig_sp = go.Figure()
        for k in model_order:
            d = model_avgs[k]
            vals = [d["h"],d["ha"],d["r"],d["h"]]
            fig_sp.add_trace(go.Scatterpolar(r=vals, theta=criteria, fill="toself",
                name=model_names[k], line=dict(color=model_colors[k],width=2), fillcolor=model_colors[k]+"18"))
        fig_sp.update_layout(**{k:v for k,v in PLOT_BASE.items() if k not in ["xaxis","yaxis"]},
            polar=dict(bgcolor="#0a0a18",
                radialaxis=dict(visible=True,range=[0,5],color="#2a2a4a",gridcolor="#141428"),
                angularaxis=dict(color="#3a3a6a",gridcolor="#141428")),
            height=340, title=dict(text="H/H/R Radar — Avg Scores",font=dict(color="#7070a0",size=12)))
        st.plotly_chart(fig_sp, use_container_width=True)

    with col_ln:
        criterion = st.radio("Criterion", ["Helpfulness","Harmlessness","Relevance"], horizontal=True)
        crit_key  = {"Helpfulness":"help","Harmlessness":"harm","Relevance":"rel"}[criterion]
        fig_ln = go.Figure()
        for k in model_order:
            fig_ln.add_trace(go.Scatter(
                x=[f"P{i+1}" for i in range(10)],
                y=MANUAL[k][crit_key],
                name=model_names[k],
                line=dict(color=model_colors[k],width=2),
                mode="lines+markers", marker=dict(size=7)))
        fig_ln.update_layout(**PLOT_BASE, height=340,
            title=dict(text=f"{criterion} — Per Prompt",font=dict(color="#7070a0",size=12)),
            yaxis=dict(**PLOT_BASE["yaxis"],range=[2,5.5],title="Score"))
        st.plotly_chart(fig_ln, use_container_width=True)


# ══════════════════════════════════════════════════════════════════
# PAGE 6 — CONFIG EXPLORER
# ══════════════════════════════════════════════════════════════════
elif page == "🔬  Config Explorer":

    st.markdown("<div class='sh'><span>Config Explorer</span> — Understand the Impact of Hyperparameters</div>", unsafe_allow_html=True)

    st.markdown("""<div class="card" style="margin-bottom:20px;">
      <div class="lbl">What this page does</div>
      <div style="font-size:13px;color:#9090c0;line-height:1.7;">
        Use the sliders below to <strong style="color:#dde0f0;">simulate how LoRA & DPO hyperparameter choices affect training</strong> — based on
        patterns observed across the two SFT and two DPO trials. This is an educational visualisation,
        not live inference.
      </div>
    </div>""", unsafe_allow_html=True)

    tab_lora, tab_dpo, tab_full = st.tabs(["LoRA Config Impact", "DPO Beta Impact", "Full Config Table"])

    with tab_lora:
        col_ctrl, col_viz = st.columns([1,2])
        with col_ctrl:
            st.markdown("<div style='font-family:IBM Plex Mono,monospace;font-size:10px;color:#5050a0;letter-spacing:2px;margin-bottom:12px;'>LORA PARAMETERS</div>", unsafe_allow_html=True)
            rank = st.slider("LoRA Rank (r)", 4, 64, 8, step=4)
            n_modules = st.slider("Number of target modules", 2, 7, 2)
            epochs = st.slider("Training epochs", 1, 4, 1)
            lr = st.select_slider("Learning rate", options=["5e-5","1e-4","2e-4","5e-4"], value="2e-4")

            lr_map = {"5e-5": 0.6, "1e-4": 0.8, "2e-4": 1.0, "5e-4": 0.85}
            lr_factor = lr_map[lr]
            trainable_m = round(rank * n_modules * 2048 * 2 / 1e6, 2)
            est_bleu = round(1.34 + (rank/64)*0.3 + (n_modules/7)*0.2 + (epochs-1)*0.05 * lr_factor, 2)
            est_loss = round(1.72 - (rank/64)*0.12 - (n_modules/7)*0.06 - (epochs-1)*0.04, 4)

            st.markdown(f"""<div class="card card-accent" style="margin-top:16px;">
              <div class="lbl">Estimated Trainable Params</div>
              <div class="val" style="font-size:20px;">{trainable_m}M</div>
            </div>
            <div class="card card-green" style="margin-top:8px;">
              <div class="lbl">Predicted Avg BLEU</div>
              <div class="val" style="font-size:20px;color:#3ddc84;">{est_bleu}</div>
            </div>
            <div class="card" style="margin-top:8px;">
              <div class="lbl">Predicted Final Loss</div>
              <div class="val" style="font-size:20px;">{est_loss}</div>
            </div>""", unsafe_allow_html=True)

        with col_viz:
            # Surface showing rank vs modules vs BLEU
            rank_vals = list(range(4, 68, 4))
            mod_vals  = list(range(2, 8))
            Z = [[round(1.34 + (r/64)*0.3 + (m/7)*0.2, 2) for m in mod_vals] for r in rank_vals]

            fig_surf = go.Figure(go.Heatmap(
                z=Z, x=[f"{m} mods" for m in mod_vals], y=[f"r={r}" for r in rank_vals],
                colorscale=[[0,"#060610"],[0.4,"#1c1840"],[0.7,"#4a42a0"],[1.0,"#6c63ff"]],
                text=[[f"{v:.2f}" for v in row] for row in Z],
                texttemplate="%{text}", textfont=dict(family="IBM Plex Mono",size=9),
                showscale=True,
            ))
            # Mark current selection
            fig_surf.add_scatter(
                x=[f"{n_modules} mods"], y=[f"r={rank}"],
                mode="markers", marker=dict(color="#ff5f5f",size=14,symbol="cross"),
                name="Your selection", showlegend=True,
            )
            fig_surf.update_layout(**{k:v for k,v in PLOT_BASE.items() if k not in ["xaxis","yaxis"]},
                height=400, margin=dict(l=60,r=16,t=44,b=44),
                title=dict(text="Rank × Modules → Estimated BLEU", font=dict(color="#7070a0",size=12)),
                xaxis=dict(tickfont=dict(size=9,color="#7070a0")),
                yaxis=dict(tickfont=dict(size=9,color="#7070a0")),
                legend=dict(font=dict(size=10)),
            )
            st.plotly_chart(fig_surf, use_container_width=True)

            # BLEU across ranks for selected modules
            bleu_line = [round(1.34+(r/64)*0.3+(n_modules/7)*0.2+(epochs-1)*0.04, 2) for r in rank_vals]
            fig_line = go.Figure()
            fig_line.add_trace(go.Scatter(x=rank_vals, y=bleu_line, mode="lines+markers",
                line=dict(color="#6c63ff",width=2.5), marker=dict(size=6),
                name=f"Modules={n_modules}, Epochs={epochs}"))
            fig_line.add_vline(x=rank, line_dash="dash", line_color="#ff5f5f",
                annotation_text=f"r={rank}", annotation_font=dict(color="#ff5f5f",size=10))
            fig_line.update_layout(**PLOT_BASE, height=220,
                title=dict(text="BLEU vs LoRA Rank", font=dict(color="#7070a0",size=12)),
                yaxis=dict(**PLOT_BASE["yaxis"],title="Est. BLEU"),
                xaxis=dict(**PLOT_BASE["xaxis"],title="Rank (r)"),
                showlegend=False, margin=dict(l=44,r=16,t=40,b=40))
            st.plotly_chart(fig_line, use_container_width=True)

    with tab_dpo:
        col_c, col_v = st.columns([1,2])
        with col_c:
            st.markdown("<div style='font-family:IBM Plex Mono,monospace;font-size:10px;color:#5050a0;letter-spacing:2px;margin-bottom:12px;'>DPO PARAMETERS</div>", unsafe_allow_html=True)
            beta = st.slider("Beta (β)", 0.01, 1.0, 0.1, step=0.01)
            dpo_rank = st.slider("DPO LoRA Rank", 4, 64, 16, step=4)
            dpo_epochs = st.slider("DPO Epochs", 1, 4, 1)

            est_margin  = round(10 + beta*18 + (dpo_rank/64)*4, 2)
            est_acc     = round(min(0.97 + beta*0.03 + dpo_epochs*0.005, 0.9999), 4)
            est_quality = round(min(3.8 + beta*1.8 + (dpo_rank/64)*0.4, 5.0), 2)
            drift_risk  = "High" if beta < 0.2 else ("Medium" if beta < 0.4 else "Low")
            drift_color = "#ff5f5f" if beta < 0.2 else ("#ffb340" if beta < 0.4 else "#3ddc84")

            st.markdown(f"""<div class="card card-accent" style="margin-top:16px;">
              <div class="lbl">Estimated Reward Margin</div>
              <div class="val" style="font-size:20px;">{est_margin:.2f}</div>
            </div>
            <div class="card card-green" style="margin-top:8px;">
              <div class="lbl">Predicted Accuracy</div>
              <div class="val" style="font-size:20px;color:#3ddc84;">{est_acc:.4f}</div>
            </div>
            <div class="card" style="margin-top:8px;">
              <div class="lbl">Policy Drift Risk</div>
              <div class="val" style="font-size:20px;color:{drift_color};">{drift_risk}</div>
            </div>
            <div class="card" style="margin-top:8px;">
              <div class="lbl">Predicted Manual Quality</div>
              <div class="val" style="font-size:20px;color:#6c63ff;">{est_quality:.2f}/5</div>
            </div>""", unsafe_allow_html=True)

        with col_v:
            beta_range = [round(x/100,2) for x in range(1, 101, 2)]
            margin_line  = [round(10 + b*18 + (dpo_rank/64)*4, 2) for b in beta_range]
            quality_line = [round(min(3.8 + b*1.8 + (dpo_rank/64)*0.4, 5.0), 2) for b in beta_range]
            drift_line   = [round(max(0, 1 - b*1.5), 2) for b in beta_range]

            fig_b = make_subplots(rows=2, cols=1, subplot_titles=["Reward Margin vs β","Manual Quality Score vs β"])
            fig_b.add_trace(go.Scatter(x=beta_range, y=margin_line, name="Reward Margin",
                line=dict(color="#6c63ff",width=2.5)), row=1, col=1)
            fig_b.add_vline(x=beta, line_dash="dash", line_color="#ff5f5f", row=1, col=1)
            fig_b.add_trace(go.Scatter(x=beta_range, y=quality_line, name="Quality Score",
                line=dict(color="#3ddc84",width=2.5)), row=2, col=1)
            fig_b.add_trace(go.Scatter(x=beta_range, y=drift_line, name="Drift Risk",
                line=dict(color="#ff5f5f",width=1.5,dash="dot"),fill="tozeroy",fillcolor="#ff5f5f10"), row=2, col=1)
            fig_b.add_vline(x=beta, line_dash="dash", line_color="#ff5f5f", row=2, col=1)
            fig_b.update_layout(**PLOT_BASE, height=480, showlegend=True,
                title=dict(text="Impact of β on DPO Training", font=dict(color="#7070a0",size=12)))
            fig_b.update_xaxes(gridcolor="#141428",linecolor="#1c1c38",title_text="Beta (β)")
            fig_b.update_yaxes(gridcolor="#141428",linecolor="#1c1c38")
            st.plotly_chart(fig_b, use_container_width=True)

    with tab_full:
        st.markdown("<div class='sh' style='margin-top:0;'>Full <span>Configuration Table</span></div>", unsafe_allow_html=True)
        cfg_df = pd.DataFrame({
            "Model":         ["Base","SFT T1","SFT T2 ★","DPO T1","DPO T2 ★"],
            "Method":        ["—","LoRA","qLoRA","DPO+LoRA","DPO+qLoRA"],
            "Rank (r)":      ["—","8","32","16","32"],
            "LoRA Alpha":    ["—","16","64","32","64"],
            "Modules":       ["—","q,v","All 7","q,k,v,o","All 7"],
            "Epochs":        ["—","1","2","1","2"],
            "Learning Rate": ["—","2e-4","1e-4","5e-5","2e-5"],
            "Beta (β)":      ["—","—","—","0.1","0.5"],
            "Trainable":     ["—","1.13M","25.2M","4.51M","25.2M"],
            "Avg BLEU":      [str(round(np.mean(BLEU_SCORES[k]),2)) for k in MODEL_META],
            "Manual Score":  ["—","—","4.23","4.40","4.90"],
        })
        st.dataframe(cfg_df.set_index("Model"), use_container_width=True)

        st.markdown("<div class='sh' style='margin-top:24px;'>Pipeline <span>Architecture</span></div>", unsafe_allow_html=True)
        steps = [
            ("1","Base Model", "TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T", "#6c63ff"),
            ("2","SFT Dataset","Dolly-15k · 5,000 train samples · Alpaca format","#38e8ff"),
            ("3","SFT Trial 1","LoRA r=8 · q,v projections · 1 epoch · lr=2e-4","#6c63ff"),
            ("4","SFT Trial 2","qLoRA r=32 · All 7 projections · 2 epochs · lr=1e-4  ← BEST SFT","#38e8ff"),
            ("5","DPO Dataset","Intel/orca_dpo_pairs · 2,000 train samples · prompt/chosen/rejected","#ffb340"),
            ("6","DPO Trial 1","β=0.1 · r=16 · 1 epoch · lr=5e-5","#ffb340"),
            ("7","DPO Trial 2","β=0.5 · r=32 · 2 epochs · lr=2e-5  ← BEST DPO","#3ddc84"),
            ("8","Evaluation", "BLEU (sacrebleu) + Manual H/H/R on 10 OOD prompts","#3ddc84"),
        ]
        for num, title, detail, color in steps:
            st.markdown(f"""<div style="display:flex;align-items:flex-start;gap:16px;padding:10px 0;border-bottom:1px solid #141428;">
              <div style="font-family:'IBM Plex Mono',monospace;font-size:12px;color:{color};
                          background:#141428;border:1px solid {color}40;border-radius:4px;
                          padding:3px 8px;min-width:28px;text-align:center;">{num}</div>
              <div>
                <div style="font-family:'Plus Jakarta Sans',sans-serif;font-size:13px;font-weight:700;color:#dde0f0;">{title}</div>
                <div style="font-family:'IBM Plex Mono',monospace;font-size:11px;color:#5050a0;margin-top:2px;">{detail}</div>
              </div>
            </div>""", unsafe_allow_html=True)
