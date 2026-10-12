# CLM vs leaky CLM vs MLM

Same architecture (d_model 128, 4 layers, 4 heads, context 128), same data, 1500 steps each.

| model | attention mask | objective val loss | honest next-token loss | hidden-char loss (pos 64) |
|---|---|---|---|---|
| CLM (causal mask) | causal | 1.8057 | 1.6943 | 1.7617 |
| leaky CLM (no mask) | none | 0.0185 | 2.3235 | n/a |
| MLM (bidirectional) | none | 1.9454 | n/a | 1.7796 |

*objective val loss* = each model's own training objective on validation text.  
*honest next-token loss* = predict the character after a 128-char prefix with no future input (the generation setting).  
*hidden-char loss* = predict a character in the middle of a window: MLM sees both sides, CLM only the left side.

**CLM (causal mask)** - left-to-right sample:

```text
ROMEO:
Say her heave and in a should me?

ABOLUNT:
When you hearts was drucking be for ster of with brands,
I will me she of thyse thou had have is a and feaver see arl
Not state the pake is laward not be w
```

**leaky CLM (no mask)** - left-to-right sample:

```text
ROMEO:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::W:




W::
I m   i: t aheerencer ser are eake tharthith tharishlararyhail sean
```

**MLM (bidirectional)** - left-to-right sample:

```text
ROMEO:
Sar II moed I am oim ab you y.
Dy ithte ing ine men oug' I
S:
Gourcini seve fot ide is ub ang
A: IOTOI YY INEBNA
D:
ASSH I mth rand aht adil ,dah I
:
GICCILUVLAR
Nakl gaant ith amer dhlang

IT:
AMNIU
```
