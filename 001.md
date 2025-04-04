# Lesson 1

## Sources
- Video ([01:22:55](https://www.youtube.com/watch?v=8SF_h3xF3cE&list=PLfYUBJiXbdtSvpQjSnJJ_PmDQB_VyT5iU&index=1)): "Practical Deep Learning for Coders: Lesson 1". Published Jul 21, 2022. (Version 5 of the course)
- Chapter: [01_intro.ipynb](https://github.com/fastai/fastbook/blob/master/01_intro.ipynb)
- [Kaggle](https://www.kaggle.com/code/jhoward/is-it-a-bird-creating-a-model-from-your-own-data): "Is it a bird? Creating a model from your own data" by Jeremy Howard

## Raw notes on the video
- [00:00:50] - In 2015, it was nearly impossible to check if a photo was of a bird; today (2022), he built that system for free in 2 minutes.
- [00:02:20] - Pix Spy, software that allows to detect the exact color on each pixel of an image. Q: FOSS alternatives?
- [00:03:10] - Images are made of numbers; pixels, each one made of numbers from 0 to 255, which indicate the amount of red, green and blue on each pixel. These are the input to the program, allowing it to figure out whether the image is about a bird, or not.

```python
# [00:01:18]
from fastbook import *               

# Search on DuckDuckGo for images of birds photos and grab 1
urls = search_images_ddg('bird photos', max_images=1)

# Print the associated URL 
len(urls),urls[0]

# [00:01:48] - Download the image and view it
dest = Path('bird.jpg')
if not dest.exists(): download_url(urls[0], dest, show_progress=False)
im = Image.open(dest)
im.to_thumb(256,256)

# [00:03:23]
# One can search for images of "birds" but not for "not birds", so he searches for "forests". He explains that he resizes the images to 400px on each side because he doesn't need particularly big ones, and it takes computers a large amount of time to open images. 
# Q: How big is "big"? When would we use bigger ones?
searches = 'forest','bird'
path = Path('bird_or_not')

if not path.exists():
    for o in searches:
        dest = (path/o)
        dest.mkdir(exist_ok=True)
        results = search_images_ddg(f'{o} photo')
        download_images(dest, urls=results[:200])
        resize_images(dest, max_size=400, dest=dest)


# If the model is trained with broken images, it won't work
# so he deletes them.
failed = verify_images(get_image_files(path))
failed.map(Path.unlink);


# [00:04:26]
# Creates a DataBlock. A DataBlock provides the fastai library all the information that it needs to create a computer vision model.
dls = DataBlock(
    blocks=(ImageBlock, CategoryBlock),
    get_items=get_image_files,   # Pass the images just downloaded
    splitter=RandomSplitter(valid_pct=0.2, seed=42),
    get_y=parent_label,
    item_tfms=[Resize(192, method='squish')]
).dataloaders(path)

dls.show_batch(max_n=6)          # Show me some images

# It's easy to check the data because we only have to see it.
# This is not the case with other type of models.



# Runs through every photo and learns "a bit more" of what forests and birds look like. In under 30 seconds he trained the model.
learn = cnn_learner(dls, resnet18, metrics=error_rate)
learn.fine_tune(3)


# Passes the first image of "bird" he downloaded at the start
is_bird,_,probs = learn.predict(PILImage.create('bird.jpg'))
print(f"This is a: {is_bird}.")
print(f"Probability it's a bird: {probs[0]:.4f}")
# Result: 1.0000
```

[00:06:20] - "[...] creating really interesting real working programs with deep learning is something that doesn't take a lot of code, didn't take any math, didn't take more than my laptop computer."

[00:11:50] - Ethical considerations https://ethics.fast.ai

[00:12:10] - AI researcher at the University of Queensland and fast.ai; homeschooling primary school teacher, so he studies education. He says Dylan Wiliam uses a system of coloured cups over the table to know if students are following along: green (fine), yellow (not quite sure), red (no idea). JH uses an online version (cups.fast.ai/fast).

[00:14:47] - Student Radek Osmulski got a job at NVIDIA AI. "[..] fast.ai alumni around the world, very frequently, like every day or two, email me to say that they've got their dream job. If you're looking for inspiration on how to get into the field... Nothing would be better than checking out Radek's work." Radek wrote a book about his journey, lots of tips about how to take advantage of fast.ai. "Meta Learning: How To Learn Deep Learning And Thrive In The Digital World".

JH highlights he started the course training the model, and not by an in-depth review of linear algebra and calculus, because he reads researchers on education like Paul Lockhart (book: "Mathematician's Lament") and David Perkins (book: "Making Learning Whole"), who talk about how much better people learn when they learn with a context in place. Most people learn effectively like the way we teach sports (show the game, how much fun it is, gradually puting more pieces together). 

"You will go into as much depth as the most sophisticated, technically detailed classses you'll find - later. But first you'll learn to be very very good at actually building and deploying models. And you will learn why and how things work as you need to, to get to the next level."

[00:18:00] - JH and friend Sylvain Gugger wrote "Deep Learning for Coders with Fastai and PyTorch: AI Applications Without a PhD". This course is based on that book. He says no material from the book is used directly because it is best to hear the same thing in multiple different ways. He wants us to read the book. (A note on YouTube says that the entire book is available for free at github.com/fastai/fastbook). The videos provide the same information presented in a different way.

Some of the people who liked the book and left reviews:
 - "[...]Don't let those PhDs have all the fun---you too can use deep learning to solve practical problems." - Hal Varian, Chief Economist @ Google
 - "[...]this is one of the best sources for a programmer to become proficient in deep learning." - Peter Norvig, Director of Research @ Google

[00:19:10] - About fastai and Jeremy Howard:
    - ~30 years learning and working in ML, including "a number of companies that relied on it".
    - Hightest ranked competitor in the world in ML competitions on Kaggle.
    - Built a company called Enlitic, which was the first to specialize in deep learning for medicine, and voted by MIT one of the 50 smartest companies in 2016, along Facebook and SpaceX.
    - He started fastai with Rachel Thomas a few years ago, which had a big impact on the world: [students won the DAWNBench competition](https://www.theverge.com/2018/5/7/17316010/fast-ai-speed-test-stanford-dawnbench-google-intel) by training big neural networks faster and cheaper than anybody in the world. This made [Google](https://cloudplatform.googleblog.com/2018/06/Cloud-TPU-now-offers-preemptible-pricing-and-global-availability.html) and [Nvidia](https://developer.nvidia.com/blog/tensor-core-ai-performance-milestones/) to start using their "special approaches" in their models.
    - Invented the "ULMFiT algorithm, which according to the Transformers book, was one of the two key foundations behind the modern NLP revolution." The paper is called "Universal language model fine-tuning for text-classification", which he invented for lesson 4 of the course (2016?).

[00:21:28] - The first version of the course was already getting Harvard attention. Quote on the screen: "fast.ai.. can actually get smart, motivated students to the point of being able to create industrial-grade ML deployments" - Harvard Business Review, The Business of Artificial Intelligence.

[00:21:40] - "[..-]it's amazing to see how many alumni have gone from this to really doing amazing things. For example, Andrej Karpathy told me that at Tesla, I think she said, pretty much everybody who joins Tesla in AI is meant to do this course. I believe at OpenAI, they told me that all the residents joining there first do this course."

[00:22:30] - Why can he build a bird recognizer in under 2 minutes and why before this was considered almost impossible? How image recognition was done in 2012: mentions a successful Stanford project that used a "classic ML approach" ("in this case, logistic regression") to predict breast cancer survival by manually creating thousands of features, wich required a cross-disciplinary group of experts (mathematicians, computer scientists, pathologists), a lot of code, a lot of years and a lot of math. Senior author Daphne Koller didn't consider deep learning because it wasn't on the radar at that time. 

"The difference with neural networks is neural networks don't require us to build these features. They build them for us. And so what actually happened was, in I think it was 2015, Matt Zeiler and Rob Fergus took a trained neural network and they looked inside it to see what it had learned. So we don't give it features, we ask it to learn features. When they looked inside a neural network, they looked at the actual weights in the model and they drew a picture of them. And this (00:24:42) was nine of the sets of weights they found."

[00:30:20] - "TensorFlow is dying in popularity in recent years[...] PyTorch is actually growing[...]" 