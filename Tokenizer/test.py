
from BPE import BasicTokenizer

text = """
"Cornerstone" is a plain word, not flashy, but it can accurately convey our emotions and confidence in the grand edifice of Chinese science fiction under construction. Therefore, we use it as the name of this original series of books.
The past ten years have been a decade of rapid development in science fiction creation. A large number of science fiction writers such as Wang Jinkang, Liu Cixin, He Hongwei, and Han Song have published a large number of science fiction masterpieces that are deeply loved by readers and have great pioneering and exploratory value. The leading science fiction magazine has grown from a traditional "Science Fiction World" to a series of publications covering all levels of readers. At the same time, the market environment for science fiction literature has also improved, and large bookstores in provincial capital cities finally have a territory for science fiction.
There are still people who often ask about the gap between Chinese science fiction and American science fiction, but the answer is now different from ten years ago. In many works (they are no longer those naive stories with no literary skills and colors, and limited imagination), this comparison has become a matter of their steak versus our potato beef. The gap is obvious—more accurately, it should be called a "difference"—but it is no longer possible to rank them. Taste issues have practical significance, which is a sign of the maturity of our science fiction.
The gap with American science fiction is actually a gap in the degree of marketization. American science fiction has formed a complete industrial chain from magazines to books to movies and then to games and toys, with full momentum; while our book publishing is still in such a situation: readers' reading needs cannot be met, while publishers lament the mere thousands of copies of science fiction books. As a result, we basically only have science fiction writers who create for love, and few who create for royalties. This is not a situation that responsible publishers are happy to see.
As the most influential professional science fiction publishing institution in China, Science Fiction World has been committed to the all-round promotion of Chinese science fiction. Science fiction book publishing is one of the key points. Chinese science fiction needs a long-term vision. It needs a pragmatic spirit and the introduction of more market-oriented methods. Therefore, we focus on the long-term, and the starting point is a piece of "cornerstone."
It needs to be specially explained that we have no restrictions on the cornerstone. Because, to build a grand edifice, various kinds of stones are needed.
We are full of expectations for such a grand edifice.
"Three-Body" can finally meet with science fiction friends, and no one expected it to be serialized in this way, which is also a helpless move. I had carefully discussed the subject matter with the editors before and felt that there was no problem, but I did not expect that this year is the 30th anniversary of the Cultural Revolution, and the single volume could not be published for a while, so it can only be like this.
In fact, this book is not about the Cultural Revolution, the content of the Cultural Revolution accounts for less than one-tenth of it, but it is a spiritual ghost that floats in the story and cannot be dispelled.
Although this book is not a sequel to "Ball Lightning," it can be regarded as a continuation of the world where that story took place. The physicist appears in the story but is no longer important, and the other people have disappeared forever. Lin Yun is really dead, although I sometimes think, if she survived, would she look like the protagonist in the end?
This is the first part of a series tentatively named "Remembrance of Earth's Past," which can be regarded as the beginning of a longer story.
This is a story about betrayal, and also a story about survival and death. Sometimes, compared to survival or death, loyalty and betrayal may be more of an issue.
What kind of power will madness and paranoia ultimately alienate within human civilization? How will the cold starry sky interrogate the morality in the heart?
The author tries to tell a modern Chinese history reinterpreted on the scale of light years, telling the legend of a civilization's two hundred destructions and rebirths.
Friends will see that the first issue of the serialization is almost not science fiction, but this book is not what this issue shows. It is not realistic science fiction, it is more ethereal than "Ball Lightning," and I hope you can patiently read on, the story will change greatly later.
In the future, readers will walk through the spiritual journey I have walked in the past year. Frankly speaking, I don't know what you will see on this dark and strange path, and I am very uneasy. But science fiction has come to this day, and it is fate to be able to walk such a long way with everyone.
"""

# text = "今天天气真好"

bt = BasicTokenizer()
bt.train(text, 200)

test_text = "tom"
e = bt.encode(test_text)
print(f"encode {test_text} to {e}")

t = bt.decode(e)
print(f"decode {e} to {t}")
