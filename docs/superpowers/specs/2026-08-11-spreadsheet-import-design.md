# Spreadsheet Import Design

## Goal

Extend the knowledge-library upload flow to accept Excel workbooks and CSV files alongside the existing PDF and TXT formats. The imported table content must enter the same RAG ingestion and retrieval pipeline as other documents.

## Supported Formats

- `.xlsx`: parsed with `openpyxl`.
- `.xls`: parsed with `xlrd`.
- `.csv`: parsed with Python's standard `csv` module.

The upload picker and its empty-state guidance will show all supported formats: PDF, TXT, XLSX, XLS, and CSV.

## Ingestion Flow

1. The upload API stores each selected file in the existing uploads directory.
2. The file-text extractor chooses a parser based on the extension.
3. Every non-empty spreadsheet row becomes a readable record: `Sheet: <name>; <header>: <value>; ...`.
4. Each workbook sheet is labelled in the generated text so retrieved answers retain their source context.
5. The generated text is passed unchanged into the existing chunking, embedding, FAISS, BM25, and source-listing flow.

CSV records are labelled as `Sheet: CSV` for the same consistent representation.

## Parsing and Errors

- CSV decoding tries UTF-8 with BOM, UTF-8, then GBK.
- Empty rows are skipped; missing column headers receive stable generated names such as `Column 1`.
- Empty workbooks, empty CSV files, unreadable encodings, corrupt files, or unsupported types cause a clear per-file upload error.
- Failed files must not be indexed as empty documents. Valid files in the same multi-file upload continue to be processed.

## Dependencies and Scope

Add `openpyxl` and `xlrd` to the backend dependencies. Do not add pandas or an office-suite runtime. Existing PDF and TXT behaviour, source metadata, deletion behaviour, and RAG retrieval logic remain unchanged.

## Verification

- Unit-test extraction for XLSX with multiple sheets, legacy XLS, UTF-8 CSV, and GBK CSV.
- Test blank rows and blank headers.
- Test that malformed or empty table files return a clear error and add no indexed chunks.
- Build the frontend after updating the accepted-file list and user-facing format guidance.
