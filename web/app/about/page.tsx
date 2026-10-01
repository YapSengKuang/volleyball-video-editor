import { InfoPage } from "../info-page";

export default function AboutPage() {
  return (
    <InfoPage title="About">
      <p>
        Rally cut turns a volleyball recording into separate rally clips. You upload one game, mark your court, and
        download the points without the waiting in between.
      </p>
      <p>
        It is built for a single fixed camera and 6v6. Movement outside the court you mark is ignored, so a game on
        the next court is left out. You can trim any clip before you export.
      </p>
    </InfoPage>
  );
}
