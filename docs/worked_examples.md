# Worked examples - exactly what the model sees

- pool: plan `smoke` manifests; renderer version pinned in each run record
- prompts below are rendered by the same code path the runner uses
- gold answers are shown for review only and never enter prompts
- generation outputs are real prediction records when present; otherwise marked **pending**

## MuSiQue local evidence task

#### 2hop__10001_33297:s1

- task: `musique_local`; kind: `positive`; group: `2hop__10001_33297`
- question syntax: natural; is_terminal: True
- gold alias source: parent_answer_aliases

Prompt the model sees (after chat-template application):

```text
Answer the question below using only the provided information.
Give a short answer. If the provided information is insufficient to answer the question, output exactly NOT FOUND.
In questions, an expression like "X >> relation" asks for that relation of X.
Information:

Paragraph 1: NSPCL Bhilai Power Plant: The NSPCL Bhilai Power Plant is a coal-fired captive power station at Bhilai in Durg district, Chhattisgarh, India. The power station owned and operated by NSPCL is a 50:50 joint venture company of NTPC Limited and SAIL to generate power for captive purposes of various steel plants owned by SAIL.

Paragraph 2: Division of Korea: A Soviet - US Joint Commission met in 1946 and 1947 to work towards a unified administration, but failed to make progress due to increasing Cold War antagonism and to Korean opposition to the trusteeship. Meanwhile, the division between the two zones deepened. The difference in policy between the occupying powers led to a polarization of politics, and a transfer of population between North and South. In May 1946 it was made illegal to cross the 38th parallel without a permit.

Paragraph 3: Group (mathematics): For any prime number p, there is also the multiplicative group of integers modulo p. Its elements are the integers 1 to p − 1. The group operation is multiplication modulo p. That is, the usual product is divided by p and the remainder of this division is the result of modular multiplication. For example, if p = 5, there are four group elements 1, 2, 3, 4. In this group, 4 · 4 = 1, because the usual product 16 is equivalent to 1, which divided by 5 yields a remainder of 1. for 5 divides 16 − 1 = 15, denoted

Paragraph 4: Dennard scaling: Dennard scaling, also known as MOSFET scaling, is a scaling law based on a 1974 paper co-authored by Robert H. Dennard, after whom it is named. Originally formulated for MOSFETs, it states, roughly, that as transistors get smaller, their power density stays constant, so that the power use stays in proportion with area; both voltage and current scale (downward) with length.

Paragraph 5: Modern history: On December 7, 1941, Japan attacked the United States at Pearl Harbor, bringing it too into the war on the Allied side. China also joined the Allies, as eventually did most of the rest of the world. China was in turmoil at the time, and attacked Japanese armies through guerilla-type warfare. By the beginning of 1942, the major combatants were aligned as follows: the British Commonwealth, the United States, and the Soviet Union were fighting Germany and Italy; and the British Commonwealth, China, and the United States were fighting Japan. The United Kingdom, the United States, the Soviet Union and China were referred as a "trusteeship of the powerful" during the World War II  and were recognized as the Allied "Big Four" in Declaration by United Nations These four countries were considered as the "Four Policemen" or "Four Sheriffs" of the Allies power and primary victors of World War II. From then through August 1945, battles raged across all of Europe, in the North Atlantic Ocean, across North Africa, throughout Southeast Asia, throughout China, across the Pacific Ocean and in the air over Japan.

Question: What did The The United Nations name "The Trusteeship Of The Powerful"

Answer:
```

Gold (never shown to the model):

- expected answer(s): ['"Four Policemen"', 'Four Policemen']
- supporting paragraph idx (in source record): 4
- prerequisite answers substituted into the question: [[1, 'The United Nations']]

Generation record: `predictions\smoke\deepseek-r1-distill-qwen-1.5b\musique_local\0\2hop__10001_33297_s1.json`

```text
NOT FOUND
```
(config `deepseek-r1-distill-qwen-1.5b`, seed 0, finish_reason `eos`)

#### 2hop__10001_72062:s1

- task: `musique_local`; kind: `positive`; group: `2hop__10001_72062`
- question syntax: natural; is_terminal: True
- gold alias source: parent_answer_aliases

Prompt the model sees (after chat-template application):

```text
Answer the question below using only the provided information.
Give a short answer. If the provided information is insufficient to answer the question, output exactly NOT FOUND.
In questions, an expression like "X >> relation" asks for that relation of X.
Information:

Paragraph 1: Ohio State Highway Patrol: Ohio State Highway Patrol Abbreviation OSHP Patch of the Ohio State Highway Patrol Badge of the Ohio State Highway Patrol Agency overview Formed 1933 Employees 2,521 (as of 2012) Legal personality Governmental: Government agency Jurisdictional structure Operations jurisdiction * State of Ohio, USA Ohio State Highway Patrol districts and posts Size 44,825 square miles (116,100 km) Population 11,538,504 (2010 Census) General nature Law enforcement Civilian police Operational structure Headquarters Columbus, Ohio Troopers 1,530 (as of 2012) Civilians 991 (as of 2011) Agency executive Colonel Paul A. Pride, Superintendent Parent agency Ohio Department of Public Safety Districts 9 Website http://statepatrol.ohio.gov/ Footnotes * Divisional agency: Division of the country, over which the agency has usual operational jurisdiction.

Paragraph 2: Public Health Agency of Canada: The Public Health Agency of Canada (French: Agence de la santé publique du Canada) is an agency of the Government of Canada that is responsible for public health, emergency preparedness, and response and infectious and chronic disease control and prevention. It was formed by Order in Council in 2004 and subsequently by legislation that came into force December 15, 2006 and it is member of the Federal Health Portfolio (along with Health Canada, the Canadian Institute of Health Research, and other organizations).

Paragraph 3: Virginia Hall: Virginia Hall Goillot DSC Croix de Guerre (6 April 1906 – 8 July 1982) was an American spy with the British Special Operations Executive during World War II and later with the American Office of Strategic Services and the Special Activities Division of the Central Intelligence Agency. She was known by many aliases, including "Marie Monin", "Germaine", "Diane", "Marie of Lyon", "Camille", and "Nicolas". The Germans gave her the nickname "Artemis". The Gestapo reportedly considered her "the most dangerous of all Allied spies".

Paragraph 4: Federal Medical Center, Butner: The Federal Medical Center, Butner (FMC Butner) is a United States federal prison in North Carolina for male inmates of all security levels who have special health needs. It is part of the Butner Federal Correctional Complex and is operated by the Federal Bureau of Prisons, a division of the United States Department of Justice. An adjacent satellite prison camp houses minimum-security male inmates.

Paragraph 5: List of specialized agencies of the United Nations: The World Health Organization (WHO) acts as a coordinating authority on international public health and deals with health and sanitation and diseases and sends medical teams to help combat epidemics. Established on 7 April 1948, the agency inherited the mandate and resources of its predecessor, the Health Organization, which had been an agency of the League of Nations. It was established in April 7, 1948 when 26 members of the United Nations ratified its Constitution. April 7 is celebrated as the World Health Day every year. The WHO is governed by 194 Member States through the World Health Assembly. Its headquarters are at Geneva in Switzerland.

Question: the specialized agency that serves as the public health division of The United Nations is

Answer:
```

Gold (never shown to the model):

- expected answer(s): ['The World Health Organization (WHO)', 'WHO', 'World Health Organization']
- supporting paragraph idx (in source record): 10
- prerequisite answers substituted into the question: [[1, 'The United Nations']]

Generation record: `predictions\smoke\deepseek-r1-distill-qwen-1.5b\musique_local\0\2hop__10001_72062_s1.json`

```text
NOT FOUND
```
(config `deepseek-r1-distill-qwen-1.5b`, seed 0, finish_reason `eos`)

#### 2hop__10001_33297:s1:neg

- task: `musique_local`; kind: `negative`; group: `2hop__10001_33297`
- question syntax: natural; is_terminal: True
- gold alias source: parent_answer_aliases

Prompt the model sees (after chat-template application):

```text
Answer the question below using only the provided information.
Give a short answer. If the provided information is insufficient to answer the question, output exactly NOT FOUND.
In questions, an expression like "X >> relation" asks for that relation of X.
Information:

Paragraph 1: NSPCL Bhilai Power Plant: The NSPCL Bhilai Power Plant is a coal-fired captive power station at Bhilai in Durg district, Chhattisgarh, India. The power station owned and operated by NSPCL is a 50:50 joint venture company of NTPC Limited and SAIL to generate power for captive purposes of various steel plants owned by SAIL.

Paragraph 2: Division of Korea: A Soviet - US Joint Commission met in 1946 and 1947 to work towards a unified administration, but failed to make progress due to increasing Cold War antagonism and to Korean opposition to the trusteeship. Meanwhile, the division between the two zones deepened. The difference in policy between the occupying powers led to a polarization of politics, and a transfer of population between North and South. In May 1946 it was made illegal to cross the 38th parallel without a permit.

Paragraph 3: Group (mathematics): For any prime number p, there is also the multiplicative group of integers modulo p. Its elements are the integers 1 to p − 1. The group operation is multiplication modulo p. That is, the usual product is divided by p and the remainder of this division is the result of modular multiplication. For example, if p = 5, there are four group elements 1, 2, 3, 4. In this group, 4 · 4 = 1, because the usual product 16 is equivalent to 1, which divided by 5 yields a remainder of 1. for 5 divides 16 − 1 = 15, denoted

Paragraph 4: Dennard scaling: Dennard scaling, also known as MOSFET scaling, is a scaling law based on a 1974 paper co-authored by Robert H. Dennard, after whom it is named. Originally formulated for MOSFETs, it states, roughly, that as transistors get smaller, their power density stays constant, so that the power use stays in proportion with area; both voltage and current scale (downward) with length.

Paragraph 5: Liapootah Power Station: The Liapootah Power Station is a run-of-the-river hydroelectric power station located in the Central Highlands region of Tasmania, Australia. The power station is situated on the Lower River Derwent catchment and is owned and operated by Hydro Tasmania.

Question: What did The The United Nations name "The Trusteeship Of The Powerful"

Answer:
```

Gold (never shown to the model):

- expected answer(s): ['"Four Policemen"', 'Four Policemen']
- supporting paragraph replaced by paragraph idx: 5
- label: unanswerable (model must output exactly NOT FOUND)

Generation record: `predictions\smoke\deepseek-r1-distill-qwen-1.5b\musique_local\0\2hop__10001_33297_s1_neg.json`

```text
NOT FOUND
```
(config `deepseek-r1-distill-qwen-1.5b`, seed 0, finish_reason `eos`)

## ProcessBench

#### gsm8k-310

- task: `processbench`; kind: `na`; group: `gsm8k:lindsay is doing the laundry, and thinks she has missed some socks. there are 50 socks that need washing. if she washes 10 pairs of socks and 15 loose socks, how many socks has lindsay missed`

Prompt the model sees (after chat-template application):

```text
The following is a math problem and a solution (split into paragraphs, enclosed with tags and indexed from 0):

[Math Problem]

Lindsay is doing the laundry, and thinks she has missed some socks. There are 50 socks that need washing. If she washes 10 pairs of socks and 15 loose socks, how many socks has Lindsay missed?

[Solution]

<paragraph_0>
Let's break down the information provided: First, the total number of socks to be washed is 50. Second, the number of pairs of socks washed is 10 pairs. Since each pair consists of 2 socks, the total number of socks in 10 pairs is 10 * 2 = 20 socks. Third, the number of loose socks washed is 15.
</paragraph_0>

<paragraph_1>
Now, we can calculate the total number of socks Lindsay has washed:
</paragraph_1>

<paragraph_2>
Total washed socks = Socks from pairs + Loose socks
Total washed socks = 20 + 15
Total washed socks = 35 socks
</paragraph_2>

<paragraph_3>
To find out how many socks Lindsay has missed, subtract the total washed socks from the total number of socks to be washed:
</paragraph_3>

<paragraph_4>
Socks missed = Total socks - Total washed socks
Socks missed = 50 - 35
Socks missed = 15 socks
</paragraph_4>

<paragraph_5>
So, Lindsay has missed 15 socks. \boxed{15}
</paragraph_5>

Your task is to review and critique the solution paragraph by paragraph. Once you identify an error in a paragraph, return the index of the paragraph where the earliest error occurs. Otherwise, return the index of -1 (which typically denotes "not found").

Please put your final answer (i.e., the index) in \boxed{}.
```

Gold (never shown to the model):

- gold: no error (the solution is correct; model must output -1)
- final answer marked correct by the authors: True

Generation record: `predictions\smoke\deepseek-r1-distill-qwen-1.5b\processbench\0\gsm8k-310.json`

```text
NOT FOUND
```
(config `deepseek-r1-distill-qwen-1.5b`, seed 0, finish_reason `eos`)

#### gsm8k-128

- task: `processbench`; kind: `na`; group: `gsm8k:a mother goes shopping. she buys cocoa at 4.20, laundry at 9.45 and a package of pasta at 1.35. she pays 20. how much change does the cashier give back`

Prompt the model sees (after chat-template application):

```text
The following is a math problem and a solution (split into paragraphs, enclosed with tags and indexed from 0):

[Math Problem]

A mother goes shopping. She buys cocoa at $4.20, laundry at $9.45 and a package of pasta at $1.35. She pays $20. How much change does the cashier give back?

[Solution]

<paragraph_0>
To find the change the cashier gives back, we need to subtract the total cost of the items from the amount the mother paid.
</paragraph_0>

<paragraph_1>
First, find the total cost of the items. Cocoa: $4.20, Laundry: $9.45, Pasta: $1.35. Total cost = $4.20 + $9.45 + $1.35 = $14.99 (round to two decimal places for money).
</paragraph_1>

<paragraph_2>
Next, subtract the total cost from the amount the mother paid. Amount paid = $20. Change = Amount paid - Total cost. Change = $20 - $14.99. Change = $5.01.
</paragraph_2>

<paragraph_3>
Therefore, the cashier gives the mother $\boxed{5.01}$ as change.
</paragraph_3>

Your task is to review and critique the solution paragraph by paragraph. Once you identify an error in a paragraph, return the index of the paragraph where the earliest error occurs. Otherwise, return the index of -1 (which typically denotes "not found").

Please put your final answer (i.e., the index) in \boxed{}.
```

Gold (never shown to the model):

- gold: first error at step 1
- final answer marked correct by the authors: False

Generation record: `predictions\smoke\deepseek-r1-distill-qwen-1.5b\processbench\0\gsm8k-128.json`

```text
NOT FOUND
```
(config `deepseek-r1-distill-qwen-1.5b`, seed 0, finish_reason `eos`)

## GSM8K (whole question, reference)

#### 0

- task: `gsm8k`; kind: `na`; group: `gsm8k:0`

Prompt the model sees (after chat-template application):

```text
Question: Janet’s ducks lay 16 eggs per day. She eats three for breakfast every morning and bakes muffins for her friends every day with four. She sells the remainder at the farmers' market daily for $2 per fresh duck egg. How much in dollars does she make every day at the farmers' market?

Solve the problem step by step, and put your final numeric answer on a line by itself in the format #### <number>.
Answer:

```

Gold (never shown to the model):

- gold: {'extracted_answer': '18', 'full_answer': 'Janet sells 16 - 3 - 4 = <<16-3-4=9>>9 duck eggs a day.\nShe makes 9 * 2 = $<<9*2=18>>18 every day at the farmer’s market.\n#### 18'}

Generation record: `predictions\smoke\deepseek-r1-distill-qwen-1.5b\gsm8k\0\0.json`

```text
NOT FOUND
```
(config `deepseek-r1-distill-qwen-1.5b`, seed 0, finish_reason `eos`)

## MATH (whole question, reference)

#### 0

- task: `math`; kind: `na`; group: `math:0`

Prompt the model sees (after chat-template application):

```text
Problem:
How many vertical asymptotes does the graph of $y=\frac{2}{x^2+x-6}$ have?

Please reason step by step, and put your final answer within \boxed{}.
Solution:

```

Gold (never shown to the model):

- gold: {'boxed_answer': '2', 'solution': 'The denominator of the rational function factors into $x^2+x-6=(x-2)(x+3)$. Since the numerator is always nonzero, there is a vertical asymptote whenever the denominator is $0$, which occurs for $x = 2$ and $x = -3$.  Therefore, the graph has $\\boxed{2}$ vertical asymptotes.', 'level': 'Level 3', 'type': 'Algebra'}

Generation record: `predictions\smoke\deepseek-r1-distill-qwen-1.5b\math\0\0.json`

```text
NOT FOUND
```
(config `deepseek-r1-distill-qwen-1.5b`, seed 0, finish_reason `eos`)

## HotpotQA distractor (whole question, reference)

#### 5a8b57f25542995d1e6f1371

- task: `hotpotqa`; kind: `na`; group: `hotpotqa:5a8b57f25542995d1e6f1371`

Prompt the model sees (after chat-template application):

```text
Answer the following question using the information in the paragraphs below. Give a short answer.

Paragraph 1: t: i

Paragraph 2: s: e

Question: Were Scott Derrickson and Ed Wood of the same nationality?
Answer:

```

Gold (never shown to the model):

- gold: {'answer': 'yes', 'supporting_facts': ['title', 'sent_id']}

Generation record: `predictions\smoke\deepseek-r1-distill-qwen-1.5b\hotpotqa\0\5a8b57f25542995d1e6f1371.json`

```text
NOT FOUND
```
(config `deepseek-r1-distill-qwen-1.5b`, seed 0, finish_reason `eos`)
