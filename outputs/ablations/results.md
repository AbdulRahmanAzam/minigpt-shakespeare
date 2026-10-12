# Ablation results

Base: `configs/ablation_base.yaml` (d_model 128, 4 layers, 4 heads, context 128, sinusoidal, 3000 steps, 8192 tokens/step). Losses and generation metrics are mean ± std over seeds [1337, 1338, 1339] (validation cross-entropy, nats/char). Step time and tokens/s come from the controlled benchmark `experiments/throughput.py` (same random batch, interleaved rounds, median), because in-run timings on a laptop GPU drift with temperature. *In-run time* is the measured training time of the seed-1337 run, which had the GPU to itself (seeds 1338/1339 ran several at a time to save wall-clock - that does not change losses).

## depth

| setting | params | best val loss | final val loss | train loss | step (ms) | tokens/s | peak mem (MB) | in-run time (s) | valid words | distinct-2 | 4-gram rep. |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 2 layers | 413,440 | 1.6301 ± 0.0049 | 1.6301 ± 0.0049 | 1.5299 ± 0.0150 | 27.0 | 303,744 | 185 | 59 | 0.876 | 0.996 | 0.000 |
| 4 layers | 809,984 | 1.5327 ± 0.0011 | 1.5327 ± 0.0011 | 1.4257 ± 0.0122 | 44.5 | 183,932 | 331 | 89 | 0.913 | 0.996 | 0.000 |
| 6 layers | 1,206,528 | 1.4947 ± 0.0056 | 1.4947 ± 0.0056 | 1.3783 ± 0.0121 | 71.4 | 114,739 | 476 | 153 | 0.937 | 0.996 | 0.000 |

## heads

| setting | params | best val loss | final val loss | train loss | step (ms) | tokens/s | peak mem (MB) | in-run time (s) | valid words | distinct-2 | 4-gram rep. |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 2 heads | 809,984 | 1.5398 ± 0.0033 | 1.5398 ± 0.0033 | 1.4348 ± 0.0124 | 44.5 | 184,191 | 275 | 82 | 0.901 | 0.996 | 0.000 |
| 4 heads | 809,984 | 1.5327 ± 0.0011 | 1.5327 ± 0.0011 | 1.4257 ± 0.0122 | 44.5 | 183,932 | 331 | 89 | 0.913 | 0.996 | 0.000 |
| 8 heads | 809,984 | 1.5476 ± 0.0076 | 1.5476 ± 0.0076 | 1.4377 ± 0.0147 | 39.5 | 207,328 | 472 | 144 | 0.905 | 0.998 | 0.000 |

## context

| setting | params | best val loss | final val loss | train loss | step (ms) | tokens/s | peak mem (MB) | in-run time (s) | valid words | distinct-2 | 4-gram rep. |
|---|---|---|---|---|---|---|---|---|---|---|---|
| ctx 64 | 809,984 | 1.5292 ± 0.0041 | 1.5292 ± 0.0041 | 1.4228 ± 0.0185 | 49.7 | 164,958 | 275 | 153 | 0.922 | 0.994 | 0.000 |
| ctx 128 | 809,984 | 1.5327 ± 0.0011 | 1.5327 ± 0.0011 | 1.4257 ± 0.0122 | 44.5 | 183,932 | 331 | 89 | 0.913 | 0.996 | 0.000 |
| ctx 256 | 809,984 | 1.5725 ± 0.0065 | 1.5725 ± 0.0065 | 1.4664 ± 0.0200 | 45.3 | 180,987 | 472 | 221 | 0.886 | 0.994 | 0.000 |

## position

| setting | params | best val loss | final val loss | train loss | step (ms) | tokens/s | peak mem (MB) | in-run time (s) | valid words | distinct-2 | 4-gram rep. |
|---|---|---|---|---|---|---|---|---|---|---|---|
| sinusoidal | 809,984 | 1.5327 ± 0.0011 | 1.5327 ± 0.0011 | 1.4257 ± 0.0122 | 44.5 | 183,932 | 331 | 89 | 0.913 | 0.996 | 0.000 |
| learned | 826,368 | 1.5514 ± 0.0048 | 1.5514 ± 0.0048 | 1.4440 ± 0.0145 | 48.7 | 168,056 | 331 | 116 | 0.911 | 0.993 | 0.000 |
| none | 809,984 | 1.7200 ± 0.0057 | 1.7200 ± 0.0057 | 1.6194 ± 0.0115 | 45.4 | 180,600 | 331 | 115 | 0.783 | 0.996 | 0.000 |

## Sample generations (seed 1, prompt 'ROMEO:')

**depth / 2 layers**

```text
ROMEO:
Yet before will and is not the dispoked you entrage--
Second hands art that us her very come.

HENRY BOLINGBROKE:
In will must of the cause heart maden in thee.

KING RICHARD III:
Give vittleman: me nor the was their in appart
His rece with a crein solent you love.
In Body's lives hope your l
```

**depth / 4 layers**

```text
ROMEO:
You before we will be love the dispoker'd age.
Still I shall have been thee us commont?

MERCUTIO:
Why, sweet not the more the own of of this blood,
To will be mine, so have late maid overs,
He will not on the seness and that this might my cousin
Of speak you love.
I cannot this good thy worl
```

**depth / 6 layers**

```text
ROMEO:
You have now own four on, but being a way,
And manner made to answer that foul hath been
Your hoar, away, my faces a wizons heavens
More in heart made did but some ensue content;
For the tears of me now on that speakly die;
The husband with a cretting hath fanture.
I cannot blave whom the sel
```

**heads / 2 heads**

```text
ROMEO:
You before we will to life the disposities of a
shrives that hangs are that us hath been to lady,
But consul. What's no prompetors or of this blood,
To will be mine the king lates that virtue,
And not and leve me, less to the tenter of him.

DUKE OF AUMERLE:
Here is not doth lives; but you do
```

**heads / 4 heads**

```text
ROMEO:
You before we will be love the dispoker'd age.
Still I shall have been thee us commont?

MERCUTIO:
Why, sweet not the more the own of of this blood,
To will be mine, so have late maid overs,
He will not on the seness and that this might my cousin
Of speak you love.
I cannot this good thy worl
```

**heads / 8 heads**

```text
ROMEO:
You before we will to lay be made to be soul:
You are my call and be this foul commonty
The lady, endrect my father prize our abses
More in heart mild did that misters, and that HereforG the heart
not him. That will lind me like up, once his burriness
With face to see and and a loss of you,
A
```

**context / ctx 64**

```text
ROMEO:
You be not well and in the times and bubelland,
That I shall have been thee us of his back:
And and sweet my faces are honour
Yours, to that I command did that misters,
I in the stail overs, and the news is wants
All and peace thee, once by at princess,
why hell not know patiently hope you, t
```

**context / ctx 128**

```text
ROMEO:
You before we will be love the dispoker'd age.
Still I shall have been thee us commont?

MERCUTIO:
Why, sweet not the more the own of of this blood,
To will be mine, so have late maid overs,
He will not on the seness and that this might my cousin
Of speak you love.
I cannot this good thy worl
```

**context / ctx 256**

```text
ROMEO:
You before will and is not the diserable
And take and my call and barrithment queen.

PAULIXENA:
And with my face, and God's fair the office come
may now my beg in entent, will is the overt,
He wife now on that speakly dispart the marnerity:
As that speak you love.
I came the gods home.

ESCA
```

**position / sinusoidal**

```text
ROMEO:
You before we will be love the dispoker'd age.
Still I shall have been thee us commont?

MERCUTIO:
Why, sweet not the more the own of of this blood,
To will be mine, so have late maid overs,
He will not on the seness and that this might my cousin
Of speak you love.
I cannot this good thy worl
```

**position / learned**

```text
ROMEO:
You have now own, which you that seek of our save?

QUEEN ELIZABETH:
Not me us are to become thou enjoy.

DUKE VINCENTIO:
No,
You shall the comme made did but soils,
Even in the Heaven's time of the news in wail!
All in you have she courted at princess,
why hell not nor of this desire
Is woul
```

**position / none**

```text
ROMEO:
You before will and is and the dispok of our take
On the day hands art that us hath bedy.

DUKE VINCENTIO:
And fallace morouse of the of in heart milend;
Which miste sengene ladist in overs, and the now
Willl be sel lin a was this me deave speak;
Sirst, why he's nour or opeselaves home.
Who t
```
