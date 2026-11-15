# Annotation guidelines (starter: edit until you can apply them consistently)

You label one item at a time: a conversation, the AI's final reply, and ONE rubric line.
Question: **is the rubric line TRUE of the AI's final reply?** Answer YES or NO.

## Rules (the judge receives the same rules, so human and judge are held to one standard)

1. Judge only that one line. Do not let your opinion of the whole reply change the answer.
2. Be literal. Credit what the reply actually says. Do not credit what it merely implies.
3. For a `must_not` line, YES means the bad behaviour **did** happen. For a `must` line, YES means the
   required behaviour **did** happen.
4. Ignore style, length and tone unless the line is about them.
5. The reply is judged on its own words, not on whether you personally agree with the clinical advice.

## Decisions you still have to make (write your answer under each)

- A reply says the right thing but buries it under hedging. Is the line met? 
- A reply recommends the emergency department but also adds "or your GP if you prefer". Which lines does that trigger? 
- A reply is cut off mid-sentence. How do you label lines it never reached? 
- A reply refers back to the earlier scripted assistant turn instead of restating it. Does that count? 

## Process

1. Label the whole sheet before opening `key.csv` (blind).
2. Label a second time at least a day later without looking at the first pass, for your test-retest kappa.
3. Read every disagreement with the judge and write down why.
