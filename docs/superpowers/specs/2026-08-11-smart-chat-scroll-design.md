# Smart Chat Scroll Design

## Goal

Replace per-update forced scrolling with stable, user-controlled follow behaviour during streamed answers.

## Behaviour

- Queue at most one scroll update per animation frame, combining multiple streamed chunks into one viewport update.
- After a question is submitted, enable automatic following.
- Continue following only while the viewport is within 96px of its bottom.
- If the user scrolls upward beyond that threshold, pause automatic following so earlier content can be read without interruption.
- Start following again on the next submitted question.
- Keep reduced-motion behaviour instantaneous and limit scrolling to the conversation viewport.

## Verification

- Confirm a long streamed answer stays at the latest content without visual jitter.
- Confirm manual upward scrolling pauses follow mode.
- Confirm the next submitted question resumes follow mode.
- Build the React frontend successfully.
