# Generation samples - minigpt_shakespeare.pt

Model: 4.77M parameters, 6 layers, 4 heads, d_model 256, context 256.  
Sampling: temperature 0.8, seed 42, 300 new characters.  
Validation loss 1.3531 (perplexity 3.87); test loss 1.5300.

## Prompt 1: `'ROMEO:'`

*valid-word rate 0.98, distinct-2 1.00, 4-gram repetition 0.00*

**Sampled (T = 0.8):**

```text
ROMEO:
Far is false; 'tis gone.

MERCUTIO:
Come, you sup your grace is that will conceive
Against me with contrives with the great England.

ROMEO:
That Hermione that we may have proved him within,
Whose canst will we are assured his daughter's mouth?

MERCUTIO:
The case, my lord, the head that send were 
```

**Greedy (T = 0), 120 characters:**

```text
ROMEO:
I will not be so much a man to be a man.

ROMEO:
I will not be so much a man to be a man.

ROMEO:
I will not be so much
```

## Prompt 2: `'JULIET:\n'`

*valid-word rate 1.00, distinct-2 0.98, 4-gram repetition 0.00*

**Sampled (T = 0.8):**

```text
JULIET:
And let us mark me kill my heart.

DUKE VINCENTIO:
So shall see that have thou seen the king.

Provost:
Ay, pity, sir; we must be spoke out at you.

Provost:
Now just, for I have sent me to do them slander.

LUCIO:
I never dream so tender: so out to see your son
only.

DUKE VINCENTIO:
Now with one o
```

**Greedy (T = 0), 120 characters:**

```text
JULIET:
What is the matter?

Nurse:
What, is it? what is the matter?

JULIET:
What, is it? what is the matter?

JULIET:
What, is
```

## Prompt 3: `'First Citizen:\n'`

*valid-word rate 0.97, distinct-2 1.00, 4-gram repetition 0.00*

**Sampled (T = 0.8):**

```text
First Citizen:
You will draw the senate is unralled aloud;
Which I am a better sour, sir, a very respect,
And seeing to do them as the that heart.

MENENIUS:
The news of our better ends can be too chatch'd
To have my plaints. To offend the oracle,
Which was true, and Rome, is my fault and live,
To you answer said.
```

**Greedy (T = 0), 120 characters:**

```text
First Citizen:
What is the consul?

CORIOLANUS:
What is the matter?

CORIOLANUS:
What is the matter?

CORIOLANUS:
What is the matter?


```

## Prompt 4: `'KING RICHARD III:\n'`

*valid-word rate 0.96, distinct-2 0.98, 4-gram repetition 0.00*

**Sampled (T = 0.8):**

```text
KING RICHARD III:
Speak the fall of this princely father's pleasure,
And fear the people of war; so he hath
To reward the Capelets and fallse, or for maid
To do wear the breath of the waning draws.

KING RICHARD III:
Nay, by cheerful flowers and the moon of the
fruit father and else would permit her company
The house
```

**Greedy (T = 0), 120 characters:**

```text
KING RICHARD III:
What is the duke?

QUEEN ELIZABETH:
The king is so far as the man that would be so.

QUEEN ELIZABETH:
And then the devil
```

## Prompt 5: `'To be, or not to be'`

*valid-word rate 0.95, distinct-2 0.98, 4-gram repetition 0.00*

**Sampled (T = 0.8):**

```text
To be, or not to be done.

CLEOMENES:
Why, belike, let's see the charter's war,
And the heart o' the firmament: in the heart
Is the law that doth appear i' the court
Which take it on't; for they mean their pleasure,
Find their thrust-bedast them have still bear it
With foul kinsman, your tongues will hack, if I
Do him
```

**Greedy (T = 0), 120 characters:**

```text
To be, or not to be the courtesy of the crown,
And therefore the common state of the crown,
Which they shall be the common that word in the
```

## Prompt 6: `'The future of artificial intelligence'`

*valid-word rate 0.95, distinct-2 0.98, 4-gram repetition 0.00*

**Sampled (T = 0.8):**

```text
The future of artificial intelligence,
To take a point more pardon of that shoes:
I saw it noble Hector
Be a true crown'd with a centre whore is now with
A little penitent coat and gave to hell.

HENRY BOLINGBROKE:
So tups the neighbour of my love intends
The crown cedit of his wit and that my shout,
Shall set the cause the reward in b
```

**Greedy (T = 0), 120 characters:**

```text
The future of artificial intelligence,
The streets of the sea and the sea of York
And the truth of the sea and the sea of York
And then the thing the common 
```
