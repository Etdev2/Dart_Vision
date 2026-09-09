# Draft enquiry — DeepDarts usage rights

> Supporting artifact for #23. **Send it, but do not sequence any work behind the reply** — see [`13-deepdarts-provenance.md`](./13-deepdarts-provenance.md) §2.5–2.6. An identical question has sat unanswered on the project's GitHub since January 2026.

## Who to contact

| Person | Role | Notes |
| --- | --- | --- |
| **William McNally** | First author; submitted the IEEE DataPort record | `wmcnally` on GitHub; ORCID linked from the DataPort record |
| **Pascale Walters** | Co-author | Named alongside Will in the open GitHub licence enquiry |

**Finding an address:** the corresponding-author email is printed on the paper itself (arXiv `2105.09880`, or the CVF open-access PDF), and ORCID profiles often list a current affiliation. Both authors were at the University of Waterloo. Do not guess an address format — take it from the paper.

**Also consider** replying on the existing thread, `wmcnally/deep-darts` issue #9, so the question is visible publicly and any answer benefits others. Low cost, and a public written answer is a good artifact to keep.

## Draft

**Subject:** DeepDarts dataset — commercial usage rights enquiry

---

Dear Dr McNally,

I'm building a single-camera dart scoring application and have been studying your CVSports 2021 paper, *DeepDarts: Modeling Keypoints as Objects for Automatic Scorekeeping in Darts using a Single Camera*. The keypoints-as-objects framing and the four-landmark homography approach are both excellent, and the dataset is a considerable piece of work — thank you for releasing it.

I'd like to be careful about rights before building anything on it, and I wasn't able to determine the position from the published material. I'd be grateful for a brief answer to four questions:

1. **Dataset — commercial use.** May a model be trained on the DeepDarts dataset and the resulting weights used in a commercial product?
2. **Attribution.** If so, how would you like the dataset and paper credited? I plan to cite the paper and the dataset DOI (`10.21227/05e7-xs69`) regardless.
3. **Code licence.** The `wmcnally/deep-darts` repository has no `LICENSE` file, so I've treated the code as all rights reserved and am reimplementing from scratch rather than reusing it. Is that intentional, or would you consider adding a permissive licence? (I notice an open licensing enquiry on the repository from January, so I may not be the only person wondering.)
4. **Derivative dataset.** If I produce a converted or curated representation of the annotations, may that be redistributed, or should I link to the IEEE DataPort record and publish only my own tooling?

A short reply on any of these would be very helpful. If the answer to (1) is no, that's entirely fine — I'd simply keep the dataset to research and benchmarking use and train the shipping model on data I've collected myself.

With thanks and best regards,

[name]
[contact]

---

## Notes on the draft

- **Four closed questions**, answerable in a sentence each. Long open-ended emails to busy academics go unanswered.
- **Offers a graceful "no."** Making refusal easy and consequence-free raises the chance of any reply at all, and a clear "no" is worth as much to us as a "yes" — it resolves the ticket either way.
- **States what we're already doing** (citing regardless, reimplementing the code) so it reads as diligence rather than a request for a favour.
- **Mentions issue #9** — signals this is a recurring question rather than one person's edge case.
- **Do not** attach a licence agreement, propose terms, or ask for a signature. That converts a two-minute reply into a legal review and guarantees silence.
