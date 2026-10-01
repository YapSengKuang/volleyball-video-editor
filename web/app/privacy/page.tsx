import { InfoPage } from "../info-page";

export default function PrivacyPage() {
  return (
    <InfoPage title="Privacy policy">
      <p>Last updated October 2026.</p>
      <p>
        Rally cut stores the video you upload, the court corners you mark, and the clip times you edit. That data is
        used only to cut your game and let you download the result.
      </p>
      <p>
        Videos are kept in private storage. They are deleted 24 hours after processing finishes. There is no account,
        and the site does not sell your footage or use it to train a public model.
      </p>
      <p>
        The server records a network address with each job so it can limit how many uploads one network can start.
        That address is not shown to other people.
      </p>
    </InfoPage>
  );
}
