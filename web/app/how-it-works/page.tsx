import { InfoPage } from "../info-page";

export default function HowItWorksPage() {
  return (
    <InfoPage title="How it works">
      <ol className="steps">
        <li>Upload an mp4 or mov, up to 8 GB.</li>
        <li>Click the four corners of your court on one frame.</li>
        <li>Review the proposed rallies. Keep, drop, or trim each clip.</li>
        <li>Download the separate clips, or one video of the whole game.</li>
      </ol>
      <p>
        The cut looks at movement inside the court you marked. Sound is only a hint. Files are deleted 24 hours after
        processing finishes.
      </p>
    </InfoPage>
  );
}
