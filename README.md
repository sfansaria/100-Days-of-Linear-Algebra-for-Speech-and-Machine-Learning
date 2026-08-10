# 100 Days of Linear Algebra for Speech and Machine Learning 🎙️📐

*Linear algebra, explained and coded — through the lens of speech.*

---

## Why this repo exists

Most linear algebra tutorials teach vectors and matrices with arrows on a whiteboard or rows of abstract numbers. That's fine, but it never quite answers the question I kept asking myself: **what does any of this have to do with the models I actually work on?**

I work on Automatic Speech Recognition and speaker diarization — systems that take a raw waveform and turn it into words, or figure out *who* said *what* in a conversation. Underneath almost every part of that pipeline — feature extraction, embeddings, attention, similarity scoring — is linear algebra doing the heavy lifting. So instead of learning the math in isolation and hoping it "clicks" later when I hit a paper or a model architecture, I decided to learn it the other way around: pick up a concept, then immediately ask *where does this show up in speech?* and code it from scratch to find out.

This repo is that experiment, one day at a time. It's part study log, part course notes, part hands-on lab — built in public, mistakes and all.

## How it works

Each day follows the same rhythm:

1. **Learn** a linear algebra concept — the intuition first, the math second.
2. **Connect** it to speech/audio ML — why it matters when you're processing waveforms, spectrograms, or embeddings.
3. **Code** it — usually from first principles before reaching for a library, so the concept actually sticks.

No day is meant to be a polished textbook chapter. It's meant to be honest: what I understood, what confused me, and the code that got me from one to the other.

## Progress so far

| Day | Topic                    | Speech/ML Angle |
|-----|--------------------------|------------------|
| 1   | Vectors                  | Representing audio frames and feature vectors |
| 2   | Vector Multiplication (Dot Product) | Measuring similarity between speaker embeddings |
| 3   | Cross Product | Building geometric intuition before moving into higher-dimensional spaces |
| 4   | Linear Transformations | How matrices reshape feature spaces — the backbone of every neural network layer |
| 5   | System of Linear Equations and Matrix Inverse | How to reconstruct clean matrix without distortions  |
| 6   | Vector Spaces, Subspace, Basis, Change of Basis | How to apply change of basis with the help of Fourier Transform |
| 7   | Orthogonality and Projections | How the mathematical concept of Orthogonality shows the relation between audio features and how clean data is extracted by projecting the distorted voice from a higher dimension to lower dimension |
| 8   | Eigen Vector and Eigen Values | How important audio features are extracted from a multidimensional dataset using PCA and covariance matrix |
| 9   | Partial Derivatives and Gradient | How model total error changes with a tweak in one specific weight and collecting all these individual partial derivatives into a single vector gives the gradient |
| 10   | Tensor                          | Representing batch_size, audio frames and feature vectors (A 3D block of numbers with different audio spectrograms packed ether to train a model simultaneously |
| 11   | Chain Rule                  |  In deep learning neural networks would be completely blind, they could calculate mistakes at the output layer but they would have no mathematical way to pass that knowledge back to update the lower tensor features without Chain Rule |
| 12   | Jacobian and Hessian Matrices                  |  Crucial for handling the non-linear, distorted paths of the human speech |
| 13   | Optimization of the Algorithms                  | Set of rules to navigate the loss landscape |

## Who this is for

- Anyone learning ML who wants the math to feel *motivated* rather than memorized.
- Speech/audio folks who want a linear algebra refresher anchored in their own domain.
- Anyone doing their own 100-day challenge and wants a companion doing the same, one topic at a time.

If you're earlier in the journey than me, welcome — read along. If you're further ahead, I'd genuinely love a pull request, an issue, or just a correction; this is a learning log, not a finished product, and I'd rather get it right than look polished.

## About me

I'm Saba — an ML engineer and Research Intern at the University of Sheffield's School of Computer Science, working on Automatic Speech Recognition and speaker diarization. This repo is where I make my own learning visible, mostly to hold myself accountable, and partly in case it helps someone else trying to connect the dots between math and speech the same way.

- GitHub: [@sfansaria](https://github.com/sfansaria)
- LinkedIn: [sabafirdausansaria-uk](https://linkedin.com/in/sabafirdausansaria-uk)

---

*Follow along, star the repo if it's useful, and feel free to open an issue if you spot an error or want to suggest a topic.*
