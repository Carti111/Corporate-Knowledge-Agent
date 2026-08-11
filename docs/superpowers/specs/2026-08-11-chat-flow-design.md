# Chat Flow Design

## Goal

Improve the React chat workspace so consecutive turns are visually distinct and newly streamed answers stay visible without manual scrolling.

## Layout

- Add a consistent 56px gap between complete conversation turns.
- Add a subtle top divider before every turn after the first, keeping the current minimal black-and-grey visual language.
- Preserve the existing user-message, answer, source, and reasoning-panel hierarchy within each turn.

## Scroll Behaviour

- When a user submits a question, scroll the conversation viewport to the newest turn.
- While answer chunks stream, follow the newest content so the cursor and text remain visible.
- Follow when sources, the reasoning plan, or the finalized saved turn appears.
- Do not use global page scrolling; only the existing conversation viewport scrolls.

## Accessibility and Verification

- Use a non-focusable end marker and `scrollIntoView` with smooth motion.
- Respect reduced-motion preferences by using instantaneous scrolling when motion is reduced.
- Verify the frontend build and confirm the chat viewport follows streamed content while preserving readable spacing between saved turns.
