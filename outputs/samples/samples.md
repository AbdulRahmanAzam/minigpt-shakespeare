# Generation samples - minigpt_shakespeare.pt

Model: 4.77M parameters, 6 layers, 4 heads, d_model 256, context 256.  
Sampling: temperature 0.8, seed 42, 300 new characters.  
Validation loss 1.3531 (perplexity 3.87); test loss 1.5300.

## Prompt 1: `'ROMEO:'`

*valid-word rate 0.98, distinct-2 0.98, 4-gram repetition 0.00*

**Sampled (T = 0.8):**

```text
ROMEO:
As foul shadows not be great as to say,
As much as your mother like a common to death.
God grant you, to this uncle is commended:
I am not like a little fall for the prince,
Which is but unite as a day as courte hand.
Farewell: no, farewell. If thou dost the abuse
As with the outward be when also, 
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

*valid-word rate 0.97, distinct-2 1.00, 4-gram repetition 0.00*

**Sampled (T = 0.8):**

```text
JULIET:
The king is a foul tender foe.

Page:
Good Paul, tell me, god my lords, did I learn,
Before I be satisfied by her driver, and yet?

BUCKINGHAM:
Why, is it not to come?

Pursuivant:
But sir, my lord, you know not a horseman in
that fast way have violent of the heavens,
Which never turns me said 'Cori
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

*valid-word rate 0.96, distinct-2 1.00, 4-gram repetition 0.00*

**Sampled (T = 0.8):**

```text
First Citizen:
Say, in the vapel of the Tower of French,
Whose shall this wander brought to the complaint
That you are but like a life and scorns.

ROMEO:
O gentleman, noise!

Second Citizen:
But she is contention like an execution.
Sirrah, good Capulet! the gods, have spoke down shed
With such shepherd's eyes, st
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

*valid-word rate 1.00, distinct-2 0.98, 4-gram repetition 0.00*

**Sampled (T = 0.8):**

```text
KING RICHARD III:
Well, faith, my lord, dear my defend Lord;
And let me this stones be too much merrily:
The did Sicilia, I'll swear.

DUKE OF AUMERLE:
Hold me to my common land,
And stand I did forswear this heart's curse;
And that through the region of many news?
That if it be made me to the bed, the store
That our
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

*valid-word rate 0.95, distinct-2 1.00, 4-gram repetition 0.00*

**Sampled (T = 0.8):**

```text
To be, or not to be true.

MENENIUS:
O, Isabel, a great word
To tell you?

MENENIUS:
That's a much of consort:
We have been kill the people:
You, look to your denies hath been in the strength,
To whose points we still be lost. Let's away
The noble tottes; for I cannot be bain
Aside myself from these sights and men's t
```

**Greedy (T = 0), 120 characters:**

```text
To be, or not to be the courtesy of the crown,
And therefore the common state of the crown,
Which they shall be the common that word in the
```

## Prompt 6: `'The future of artificial intelligence'`

*valid-word rate 0.94, distinct-2 1.00, 4-gram repetition 0.00*

**Sampled (T = 0.8):**

```text
The future of artificial intelligence,
Which once of more preparation will say
Than I should call thee to thy strange shall
With instatiate traitors, which now thou shalt have made
The bawdy crown trees beloving the earth.

GREMIO:
A reselves, for our sword, man! Who's best vow?

HORTENSIO:
He had not forgot to your short succeeding,
A
```

**Greedy (T = 0), 120 characters:**

```text
The future of artificial intelligence,
The streets of the sea and the sea of York
And the truth of the sea and the sea of York
And then the thing the common 
```
