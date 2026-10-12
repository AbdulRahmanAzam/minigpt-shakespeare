# Decoding strategy comparison

4 prompts x 3 samples x 300 characters (greedy: 1 sample per prompt).

| strategy | valid words | distinct-1 | distinct-2 | 4-gram repetition | model NLL |
|---|---|---|---|---|---|
| greedy (T=0) | 0.986 | 0.338 | 0.488 | 0.350 | 0.529 |
| T=0.5 | 0.996 | 0.700 | 0.963 | 0.000 | 0.775 |
| T=0.8 | 0.979 | 0.806 | 0.989 | 0.000 | 0.886 |
| T=1.0 | 0.950 | 0.856 | 0.993 | 0.000 | 1.043 |
| T=1.3 | 0.897 | 0.905 | 0.999 | 0.000 | 1.374 |
| T=0.8, top-k=10 | 0.972 | 0.782 | 0.988 | 0.000 | 0.893 |
| T=1.0, top-p=0.9 | 0.969 | 0.810 | 0.991 | 0.000 | 0.937 |

## Examples (prompt 'ROMEO:')

**greedy (T=0)**

```text
ROMEO:
I will not be so much a man to be a man.

ROMEO:
I will not be so much a man to be a man.

ROMEO:
I will not be so much as the sea as thou art.

ROMEO:
I am a man to be a man to be a man.

ROMEO:
I have a little service to the sea and there.

JULIET:
I have not so much a man to be a man.

JULIET:
I
```

**T=0.5**

```text
ROMEO:
I will not be a poison, sir.

ROMEO:
O, what says not she?

ROMEO:
And so that? speak not to be so?

JULIET:
And say 'tis the sun; the same is a sense;
For what this is the truth of his war,
The seal'st that I might be prevail'd,
And fill the second he is a perfect,
And both of the dearest reign an
```

**T=0.8**

```text
ROMEO:
I am I no general in his birth.

MARIANA:
For I have heard my lips in my bed happier:
He is grace to my lord dead, of thyself.

BENVOLIO:
Speak, say your servant Northumberland,
The duke and citizens so true, or your true care
Of some parliament will speak weeds from the measure,
That's not but tha
```

**T=1.0**

```text
ROMEO:
Say this earl and impossible conceit doubt?
When you hear, my lord?

BALTHASAR:
Ha! how now, my lord?
Who's my lord? but stand not the king?

FRIAR LAURENCE:
No, Nature, but that he did use him by me.
Mattude of your treasure blood, you must call unfortune
Meet urge from me to the midwife; reigning
```

**T=1.3**

```text
ROMEO:
Say things like moiments,
I will bith murder well not. Thyself thus.
By thine kisses of night, I love this thee.

CLAUDIO:
But not true a pleasant few:
Could away it stay! Stands are husbands in seven,
Which, our royal preceeder, some ill-discolour:
Let profess us ouf heaven our mother's Rome
Hath 
```

**T=0.8, top-k=10**

```text
ROMEO:
I am too see, and in a subject shall death.

BENVOLIO:
That's not for the princes of the stores.

BENVOLIO:
Then say so. What, no; the matter, and I would
swear to be ta'en to put the part bended.

ROMEO:
I'll make an appetal blood for blood; and, not fight.

ROMEO:
A gentleman to make an end of a 
```

**T=1.0, top-p=0.9**

```text
ROMEO:
Say then salt thou must be, my heart die.

BENVOLIO:
I would were content, by the swift with many stones
And deserve the stones. Here is a prisoner
To see the whole land belly put the law;
Or in the danger deep is friendly person
To make us to lose again before us off,
Now he is more and my brother
```
