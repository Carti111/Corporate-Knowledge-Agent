# Table Source Preview Design

## Goal

Keep cited spreadsheet sources readable by splitting long table content into bounded retrieval chunks and collapsing long source previews in the chat UI.

## Backend

- Split long paragraphs that exceed the configured chunk size using the existing overlap setting, including English and tabular text without Chinese sentence punctuation.
- Preserve table row context already produced by the spreadsheet extractor.
- Re-index `sample-financial.xlsx` after the chunking change so its stored source excerpts use the new bounded chunks.

## Frontend

- Limit each source excerpt to five visual lines by default.
- Provide an explicit expand/collapse control only when the excerpt overflows.
- Keep the source title, file name, and relevance score always visible.

## Verification

- Confirm the financial workbook creates several bounded chunks rather than one oversized chunk.
- Confirm a source card initially remains compact and expands on request.
- Confirm the React frontend build succeeds.
