# Local judge validation

Tested on Windows CPU on 17 November 2026 with the project's published encoder and selector, Qwen2.5-1.5B-Instruct Q4_K_M and llama.cpp b12445. Both services listened on loopback.

| Input | Trained summarizer | Judge coverage | Elapsed judge time |
| --- | --- | --- | --- |
| Fictional wage-claim example in the app | 50 words | 4 of 6 extracts | 11.6 s |
| 11-page judgment, 1976_2013_12_1501_64171_Judgement_09-Sep-2025.pdf | 800 words | 4 of 32 extracts | 18.4 s |

For the PDF the app showed support for the decision on page 11, reasoning on page 4 and issue on page 2. It withheld support for the background because the selected quotation did not contain the full extract. The test checked the quotations against the stored page text, opened the corresponding PDF page images and inspected the exported ZIP. These are machine verdicts. No legal reviewer graded this PDF during the test.

## Four fictional controls

The test source includes an earlier dismissal, a present order allowing the appeal and payment of Rs. 25,000 within six weeks. Run `python scripts/demo_local_judge.py` to repeat it. These cases were also used during development.

| Claim under review | Expected | Observed in the final control run |
| --- | --- | --- |
| Correct amount and six-week deadline | Supported | Insufficient evidence |
| Present court dismissed the appeal | Contradicted | Insufficient evidence |
| Rs. 50,000 within two weeks | Contradicted | Contradicted |
| Court awarded 12 percent interest | Insufficient evidence | Insufficient evidence |

The judge caught the changed amount and deadline in this run. It also abstained on a correct claim and missed an explicit outcome conflict. Repeated development runs produced different citations. For a faculty demonstration, open the evidence behind each verdict and use the parsed details to inspect individual amounts, deadlines and possible omissions. Measuring judge precision and recall on unseen judgments requires independently labeled claims.

## App checks

All 122 repository tests passed, including CPU training tests, citation validation, judge failure and retry, cached reports, session isolation and ZIP export. A Chromium browser test exercised the classroom example, switched all three review views, highlighted a citation, saved a reviewer draft, inspected the ZIP and cleared the workspace. The 390-pixel mobile viewport had no horizontal overflow. The browser reported no JavaScript errors during this flow.
