#!/usr/bin/env bash
# MuSiQue selection review: recommended decisions for bundles 01-04 (100 parents).
# Run from the AB_Test repo root on the machine that holds artifacts/manifests/.
# Every line is a recommendation; change any line you disagree with before running.
# Borderline items (sections 3 and 6) default to EXCLUDE; switch a line to
#   weval review approve --parent <id> [--target positive]
# to keep that item instead.
set -euo pipefail

# ---- 1. positives: approve (57) ----
weval review approve --parent 3hop1__662413_42197_18397 --target positive   # 1-02
weval review approve --parent 4hop1__60000_59747_211319_557671 --target positive   # 1-05
weval review approve --parent 3hop1__751782_2053_52946 --target positive   # 1-06
weval review approve --parent 2hop__142699_67465 --target positive   # 1-11
weval review approve --parent 3hop1__106847_160713_77246 --target positive   # 1-12
weval review approve --parent 2hop__16844_42173 --target positive   # 1-13
weval review approve --parent 2hop__256553_617289 --target positive   # 1-14
weval review approve --parent 3hop2__222979_840908_40768 --target positive   # 1-15
weval review approve --parent 2hop__59366_71206 --target positive   # 1-18
weval review approve --parent 2hop__543853_124498 --target positive   # 1-19
weval review approve --parent 2hop__107185_64006 --target positive   # 1-23
weval review approve --parent 3hop1__691197_15840_36002 --target positive   # 1-24
weval review approve --parent 3hop1__6379_42197_18397 --target positive   # 2-01
weval review approve --parent 3hop1__773623_87694_124169 --target positive   # 2-02
weval review approve --parent 2hop__847760_80026 --target positive   # 2-05
weval review approve --parent 2hop__6735_6733 --target positive   # 2-06
weval review approve --parent 4hop2__5206_39078_8987_8974 --target positive   # 2-08
weval review approve --parent 3hop2__607269_223623_162182 --target positive   # 2-09
weval review approve --parent 2hop__85753_64503 --target positive   # 2-11
weval review approve --parent 3hop2__30152_107291_20999 --target positive   # 2-12
weval review approve --parent 2hop__88372_83837 --target positive   # 2-13
weval review approve --parent 2hop__122140_37168 --target positive   # 2-14
weval review approve --parent 3hop1__843207_755530_88011 --target positive   # 2-15
weval review approve --parent 3hop1__671542_651917_47686 --target positive   # 2-17
weval review approve --parent 2hop__87589_362941 --target positive   # 2-18
weval review approve --parent 2hop__195917_575657 --target positive   # 2-20
weval review approve --parent 4hop2__567956_14670_8987_8529 --target positive   # 3-01
weval review approve --parent 2hop__384238_74735 --target positive   # 3-02
weval review approve --parent 3hop1__145427_106426_157788 --target positive   # 3-04
weval review approve --parent 3hop2__90098_19639_10557 --target positive   # 3-06
weval review approve --parent 4hop1__151650_5274_458768_33632 --target positive   # 3-07
weval review approve --parent 2hop__256778_131879 --target positive   # 3-08
weval review approve --parent 3hop2__304722_526346_63959 --target positive   # 3-11
weval review approve --parent 3hop1__131829_39743_24526 --target positive   # 3-12
weval review approve --parent 2hop__146919_29244 --target positive   # 3-13
weval review approve --parent 3hop2__127483_156850_10557 --target positive   # 3-14
weval review approve --parent 2hop__123145_66723 --target positive   # 3-16
weval review approve --parent 3hop2__326964_7855_7713 --target positive   # 3-17
weval review approve --parent 3hop2__87184_38738_76291 --target positive   # 3-18
weval review approve --parent 2hop__63214_319330 --target positive   # 3-22
weval review approve --parent 4hop3__673447_132409_145082_35031 --target positive   # 3-23
weval review approve --parent 2hop__95687_2684 --target positive   # 4-01
weval review approve --parent 2hop__814856_110882 --target positive   # 4-02
weval review approve --parent 3hop1__639955_834494_34109 --target positive   # 4-04
weval review approve --parent 3hop1__129499_33897_81096 --target positive   # 4-05
weval review approve --parent 2hop__29368_29376 --target positive   # 4-06
weval review approve --parent 3hop1__135793_87694_64412 --target positive   # 4-07
weval review approve --parent 3hop1__816887_127905_80286 --target positive   # 4-09
weval review approve --parent 2hop__703493_162341 --target positive   # 4-12
weval review approve --parent 4hop1__711307_49925_13759_736921 --target positive   # 4-13
weval review approve --parent 2hop__411091_625230 --target positive   # 4-14
weval review approve --parent 2hop__449673_91104 --target positive   # 4-17
weval review approve --parent 2hop__859431_18803 --target positive   # 4-18
weval review approve --parent 2hop__123166_47134 --target positive   # 4-19
weval review approve --parent 3hop1__106845_5188_86687 --target positive   # 4-20
weval review approve --parent 2hop__804417_126089 --target positive   # 4-23
weval review approve --parent 2hop__813461_31270 --target positive   # 4-24

# ---- 2. positives: exclude (clear problem) (29) ----
weval review exclude --parent 3hop1__106313_443779_52195 --target positive --note 'bilateral commission of Indonesia and East Timor; "Indonesia" is equally valid'   # 1-03
weval review exclude --parent 4hop1__56270_53706_795904_580996 --target positive --note 'support only gives a school address "Rio Linda, Sacramento"; no border relation stated'   # 1-04
weval review exclude --parent 4hop1__161605_32392_823060_610794 --target positive --note 'radio station licensed to Forest Acres serving Columbia market; no border relation stated'   # 1-07
weval review exclude --parent 2hop__70584_198548 --target positive --note 'film cast list; does not state Katharine Ross is Sam Elliott'"'"'s spouse'   # 1-08
weval review exclude --parent 4hop3__661103_698586_1926_54362 --target positive --note 'paragraph never says who burned Atlanta (Hood ordered assets destroyed; Sherman is the common answer); gold "Confederate Gen." prefix caps "John Bell Hood" at F1 0.75'   # 1-10
weval review exclude --parent 2hop__753260_71701 --target positive --note 'asks the year, gold is "November 5" and no year appears in the paragraph'   # 1-16
weval review exclude --parent 2hop__37656_160040 --target positive --note 'garbled question ("this stuff"); Pascal and Wagenseil also defend Christianity in the same paragraph'   # 1-17
weval review exclude --parent 4hop2__71753_73205_70784_79935 --target positive --note 'garbled question ("north of Middle East, and Persian Gulf"); paragraph says Saudi Arabia is north of Yemen'   # 1-20
weval review exclude --parent 2hop__78766_461854 --target positive --note 'miniseries synopsis; does not state Olivier is Leigh'"'"'s spouse'   # 1-22
weval review exclude --parent 3hop2__57238_1926_54362 --target positive --note 'same subquestion and support as 1-10'   # 1-25
weval review exclude --parent 4hop1__571599_49925_13759_736921 --target positive --note 'truncated question ("reform and address"); paragraph also names Hus, Waldo and Wycliffe as earlier reformers'   # 2-03
weval review exclude --parent 4hop3__152056_698586_57596_54362 --target positive --note 'discography says "from Atlanta"; place of death is not stated'   # 2-04
weval review exclude --parent 2hop__359312_18803 --target positive --note 'two April-2010 Shell announcements are visible (Finland buyer search; divesting African downstream business)'   # 2-07
weval review exclude --parent 4hop1__636860_49925_13759_736921 --target positive --note 'same subquestion and support as 2-03'   # 2-10
weval review exclude --parent 2hop__32441_274635 --target positive --note 'paragraph gives Rynning'"'"'s branch (US Army), not Roosevelt'"'"'s'   # 2-16
weval review exclude --parent 2hop__216535_370666 --target positive --note 'paragraph says a Nankai professor was born in Tianjin; the university'"'"'s location is not stated'   # 2-21
weval review exclude --parent 3hop1__848113_15840_36002 --target positive --note 'duplicate of 1-24 (same subquestion, same support paragraph)'   # 2-23
weval review exclude --parent 2hop__73921_807969 --target positive --note 'support is about a different person (Ahmed Salah Hosny); Mohamed Salah is not mentioned'   # 2-25
weval review exclude --parent 4hop1__105002_17130_70784_61381 --target positive --note 'garbled question (same as 1-20); paragraph says Saudi Arabia is north of Yemen'   # 3-05
weval review exclude --parent 2hop__60686_229757 --target positive --note 'film cast list; does not state Will Arnett is Amy Poehler'"'"'s spouse'   # 3-09
weval review exclude --parent 2hop__278022_71701 --target positive --note 'same subquestion and support as 1-16 (asks year, gold "November 5")'   # 3-15
weval review exclude --parent 3hop1__801799_547811_80702 --target positive --note 'opera staged in Venice; Vivaldi'"'"'s place of birth is not stated'   # 3-20
weval review exclude --parent 3hop2__89854_90327_76291 --target positive --note 'duplicate of 3-18 (same subquestion and support)'   # 3-25
weval review exclude --parent 3hop2__802066_223623_162182 --target positive --note 'duplicate of 2-09 (same subquestion and support)'   # 4-08
weval review exclude --parent 2hop__623931_656446 --target positive --note 'film cast list; does not state John Alderton is Pauline Collins'"'"'s spouse'   # 4-10
weval review exclude --parent 2hop__150866_11402 --target positive --note 'question has lost its subject (Atlantic City); the Pollock, Louisiana paragraph also gives a Census Bureau total area (1.3 sq mi)'   # 4-11
weval review exclude --parent 3hop1__831499_228453_10972 --target positive --note 'paragraph is about the cocktail ("a local New Orleans variation"), not the company'"'"'s headquarters'   # 4-16
weval review exclude --parent 4hop1__802394_153080_33897_81096 --target positive --note 'asks for a city, gold is a state ("Arizona"); correct answers Nogales / Los Angeles score 0'   # 4-21
weval review exclude --parent 3hop1__208108_547811_41132 --target positive --note 'same subquestion and support as 3-20 (birthplace not stated)'   # 4-25

# ---- 3. positives: BORDERLINE, default exclude (14) ----
weval review exclude --parent 3hop1__857_846_7810 --target positive --note 'borderline: gold "RMB 98,174(US$16041)" normalizes to one glued token; a correct "RMB 98,174" scores F1 0.5'   # 1-01
weval review exclude --parent 3hop1__145411_443779_52195 --target positive --note 'borderline: asks president of the newly independent country; infobox gives the later incumbent (Guterres), not the president at independence'   # 1-09
weval review exclude --parent 2hop__80546_75184 --target positive --note 'borderline: paragraph says Claudia Wells played "the character" (Jennifer) but never that she is Marty'"'"'s / Michael J. Fox'"'"'s girlfriend'   # 1-21
weval review exclude --parent 3hop1__309596_160713_77246 --target positive --note 'borderline: natural answer "Hinduism" scores F1 0 against gold "Hindu"'   # 2-19
weval review exclude --parent 2hop__451574_120065 --target positive --note 'borderline: garbled question: the town did not end; 1738 is when it stopped being the ducal residence'   # 2-22
weval review exclude --parent 3hop2__89048_228_66294 --target positive --note 'borderline: paragraph lists many influences (Diana Ross, Whitney Houston, Mariah Carey ...); any of them is a correct answer that scores 0'   # 2-24
weval review exclude --parent 4hop1__88342_49853_128008_54974 --target positive --note 'borderline: paragraph names two leagues (MLB and the American League); "American League" is equally valid and scores F1 0.4'   # 3-03
weval review exclude --parent 3hop2__43283_12039_93690 --target positive --note 'borderline: question has lost its subject (University of Houston graduates); gold "in Houston" caps the natural answer "Houston" at F1 0.67'   # 3-10
weval review exclude --parent 2hop__785711_73244 --target positive --note 'borderline: "where is the origin": the infobox gives both the peak (Thaba Putsoa) and its location (Maloti Mountains, Lesotho); "Maloti Mountains, Lesotho" is correct and scores 0'   # 3-19
weval review exclude --parent 2hop__693157_131890 --target positive --note 'borderline: paragraph names two bodies of water by Deptford (Deptford Creek and the Thames)'   # 3-21
weval review exclude --parent 4hop2__71753_648517_70784_61381 --target positive --note 'borderline: natural answer "off the coast of Kuwait in the Persian Gulf" scores F1 0.44; "Kuwait" scores 0'   # 3-24
weval review exclude --parent 2hop__358270_108549 --target positive --note 'borderline: gold lists the two developers as separate aliases, so the complete answer "Alfred Gough and Miles Millar" scores F1 0.57'   # 4-03
weval review exclude --parent 2hop__71715_50429 --target positive --note 'borderline: gold "10,310,371 (2016 est.)"; the bare correct number scores F1 0.5'   # 4-15
weval review exclude --parent 2hop__124412_402641 --target positive --note 'borderline: paragraph names six children of Chiang Ching-kuo; any of the other five is correct and scores 0'   # 4-22

# ---- 4. negatives: approve (82) ----
weval review approve --parent 3hop1__857_846_7810   # 1-01
weval review approve --parent 3hop1__662413_42197_18397   # 1-02
weval review approve --parent 3hop1__106313_443779_52195   # 1-03
weval review approve --parent 4hop1__56270_53706_795904_580996   # 1-04
weval review approve --parent 4hop1__60000_59747_211319_557671   # 1-05
weval review approve --parent 4hop1__161605_32392_823060_610794   # 1-07
weval review approve --parent 2hop__70584_198548   # 1-08
weval review approve --parent 3hop1__145411_443779_52195   # 1-09
weval review approve --parent 4hop3__661103_698586_1926_54362   # 1-10
weval review approve --parent 2hop__142699_67465   # 1-11
weval review approve --parent 2hop__256553_617289   # 1-14
weval review approve --parent 3hop2__222979_840908_40768   # 1-15
weval review approve --parent 2hop__753260_71701   # 1-16
weval review approve --parent 2hop__37656_160040   # 1-17
weval review approve --parent 2hop__59366_71206   # 1-18
weval review approve --parent 2hop__543853_124498   # 1-19
weval review approve --parent 4hop2__71753_73205_70784_79935   # 1-20
weval review approve --parent 2hop__78766_461854   # 1-22
weval review approve --parent 3hop1__691197_15840_36002   # 1-24
weval review approve --parent 3hop2__57238_1926_54362   # 1-25
weval review approve --parent 3hop1__6379_42197_18397   # 2-01
weval review approve --parent 4hop3__152056_698586_57596_54362   # 2-04
weval review approve --parent 2hop__847760_80026 --note 'hard negative: Roy Kinnear paragraph says he played Henry Salt (Veruca'"'"'s father)'   # 2-05
weval review approve --parent 2hop__6735_6733 --note 'hard negative: three other Catalan paragraphs remain, none states Central Catalan'"'"'s division'   # 2-06
weval review approve --parent 4hop2__5206_39078_8987_8974   # 2-08
weval review approve --parent 3hop2__607269_223623_162182   # 2-09
weval review approve --parent 4hop1__636860_49925_13759_736921   # 2-10
weval review approve --parent 2hop__85753_64503   # 2-11
weval review approve --parent 3hop2__30152_107291_20999   # 2-12
weval review approve --parent 2hop__88372_83837   # 2-13
weval review approve --parent 2hop__122140_37168   # 2-14
weval review approve --parent 3hop1__843207_755530_88011   # 2-15
weval review approve --parent 3hop1__671542_651917_47686   # 2-17
weval review approve --parent 2hop__87589_362941   # 2-18
weval review approve --parent 2hop__195917_575657   # 2-20
weval review approve --parent 2hop__216535_370666   # 2-21
weval review approve --parent 3hop1__848113_15840_36002   # 2-23
weval review approve --parent 3hop2__89048_228_66294   # 2-24
weval review approve --parent 2hop__73921_807969   # 2-25
weval review approve --parent 4hop2__567956_14670_8987_8529   # 3-01
weval review approve --parent 2hop__384238_74735   # 3-02
weval review approve --parent 3hop1__145427_106426_157788   # 3-04
weval review approve --parent 3hop2__90098_19639_10557   # 3-06
weval review approve --parent 2hop__256778_131879   # 3-08
weval review approve --parent 2hop__60686_229757   # 3-09
weval review approve --parent 3hop2__43283_12039_93690 --note 'hard negative: Margaret Altmann paragraph says "after graduation, she stayed in Germany"'   # 3-10
weval review approve --parent 3hop2__304722_526346_63959   # 3-11
weval review approve --parent 3hop1__131829_39743_24526   # 3-12
weval review approve --parent 2hop__146919_29244 --note 'hard negative: another Paris paragraph lists CDG, Orly and Beauvais but does not rank them'   # 3-13
weval review approve --parent 3hop2__127483_156850_10557   # 3-14
weval review approve --parent 2hop__278022_71701   # 3-15
weval review approve --parent 2hop__123145_66723   # 3-16
weval review approve --parent 3hop2__326964_7855_7713   # 3-17
weval review approve --parent 3hop2__87184_38738_76291   # 3-18
weval review approve --parent 2hop__785711_73244   # 3-19
weval review approve --parent 3hop1__801799_547811_80702   # 3-20
weval review approve --parent 2hop__693157_131890   # 3-21
weval review approve --parent 2hop__63214_319330   # 3-22
weval review approve --parent 4hop3__673447_132409_145082_35031   # 3-23
weval review approve --parent 4hop2__71753_648517_70784_61381   # 3-24
weval review approve --parent 3hop2__89854_90327_76291   # 3-25
weval review approve --parent 2hop__95687_2684   # 4-01
weval review approve --parent 2hop__814856_110882   # 4-02
weval review approve --parent 2hop__358270_108549   # 4-03
weval review approve --parent 3hop1__639955_834494_34109   # 4-04
weval review approve --parent 3hop1__129499_33897_81096 --note 'hard negative: a 2001 Indy race won by Cristiano da Matta (Surfers Paradise, not Phoenix) remains'   # 4-05
weval review approve --parent 2hop__29368_29376   # 4-06
weval review approve --parent 3hop1__135793_87694_64412   # 4-07
weval review approve --parent 3hop2__802066_223623_162182   # 4-08
weval review approve --parent 3hop1__816887_127905_80286   # 4-09
weval review approve --parent 2hop__623931_656446   # 4-10
weval review approve --parent 2hop__703493_162341   # 4-12
weval review approve --parent 4hop1__711307_49925_13759_736921   # 4-13
weval review approve --parent 2hop__411091_625230   # 4-14
weval review approve --parent 2hop__71715_50429   # 4-15
weval review approve --parent 3hop1__831499_228453_10972   # 4-16
weval review approve --parent 2hop__449673_91104   # 4-17
weval review approve --parent 2hop__123166_47134   # 4-19
weval review approve --parent 4hop1__802394_153080_33897_81096   # 4-21
weval review approve --parent 2hop__124412_402641 --note 'hard negative: Chiang Ching-kuo biography paragraph remains but names no children'   # 4-22
weval review approve --parent 2hop__804417_126089   # 4-23
weval review approve --parent 3hop1__208108_547811_41132   # 4-25

# ---- 5. negatives: exclude (a remaining paragraph answers the question) (2) ----
weval review exclude --parent 2hop__107185_64006 --note 'remaining Black Death paragraph says "about a third of the European population had already perished" - answerable'   # 1-23
weval review exclude --parent 2hop__150866_11402 --note 'remaining Pollock, Louisiana paragraph answers the subjectless question ("total area of 1.3 square miles")'   # 4-11

# ---- 6. negatives: BORDERLINE, default exclude (2) ----
weval review exclude --parent 4hop1__571599_49925_13759_736921 --note 'borderline: remaining Pope Paul VI paragraph describes Church reforms he drove; with a question this vague, NOT FOUND is not the only defensible reply'   # 2-03
weval review exclude --parent 4hop1__105002_17130_70784_61381 --note 'borderline: remaining Iran paragraph puts the Persian Gulf on Iran'"'"'s south, so "Iran" is a defensible answer to "north of ... the Persian Gulf"'   # 3-05

# 14 parents have no negative (excluded at build: numeric policy / answer visible); nothing to do for them.

weval review list --only proposed   # should list nothing for these 100 parents
# then: weval build   (decisions take effect in the manifests)
