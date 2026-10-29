/**
 * Creates the Rag to Riches participant survey as a Google Form,
 * with a linked Google Sheet for the responses.
 *
 * How to run:
 *  1. Go to https://script.google.com and click "New project".
 *  2. Paste this whole file, replacing the default code.
 *  3. Pick createWorkshopSurvey in the function dropdown and click Run.
 *  4. Accept the permission prompt (it needs Forms, Sheets and Drive access).
 *  5. Open View > Logs to get the share link and the edit link.
 */
function createWorkshopSurvey() {
  var form = FormApp.create('Rag to Riches: participant survey');
  form.setDescription(
    'Two minutes, anonymous. Your answers help us pitch the workshop at the right level.'
  );
  form.setCollectEmail(false);
  form.setAllowResponseEdits(false);
  form.setConfirmationMessage('Thanks! See you at the workshop.');

  form.addMultipleChoiceItem()
    .setTitle('Who are you?')
    .setChoiceValues(['Student', 'Working', 'Both studying and working'])
    .showOtherOption(true)
    .setRequired(true);

  form.addTextItem()
    .setTitle('Where do you study or work?')
    .setHelpText('For example UiB, HVL, NHH or a company name.')
    .setRequired(true);

  form.addMultipleChoiceItem()
    .setTitle('What do you study or do?')
    .setChoiceValues([
      'Informatics / computer science',
      'Other STEM',
      'Business / economics',
      'Humanities / social sciences',
      'Software developer',
      'Data / ML',
    ])
    .showOtherOption(true);

  form.addScaleItem()
    .setTitle('How comfortable are you with Python?')
    .setBounds(1, 5)
    .setLabels('Beginner', 'Very comfortable')
    .setRequired(true);

  form.addCheckboxItem()
    .setTitle('What is your AI background?')
    .setHelpText('Tick everything that applies.')
    .setChoiceValues([
      'Taken an AI or ML course',
      'Use ChatGPT or similar regularly',
      'Built something with an LLM API',
      'Heard of RAG',
      'Heard of MCP',
      'Used embeddings or vector search',
      'None of the above',
    ]);

  form.addScaleItem()
    .setTitle('How interested are you in AI in general?')
    .setBounds(1, 5)
    .setLabels('Slightly', 'Very')
    .setRequired(true);

  form.addCheckboxItem()
    .setTitle('What do you want to get out of this workshop?')
    .setChoiceValues([
      'Understand how AI agents work',
      'Build something for a course or project',
      'Ideas for my thesis',
      'Skills for my CV and career',
      'Meet other people interested in AI',
      'Just curious',
    ])
    .showOtherOption(true);

  form.addParagraphTextItem()
    .setTitle('Anything else you hope to learn or build?')
    .setHelpText('Optional.');

  form.addTextItem()
    .setTitle('Email, if you want to hear about future YoungAI events')
    .setHelpText('Optional.');

  var sheet = SpreadsheetApp.create('Rag to Riches survey responses');
  form.setDestination(FormApp.DestinationType.SPREADSHEET, sheet.getId());

  Logger.log('Share this link with participants: ' + form.getPublishedUrl());
  Logger.log('Edit the form here: ' + form.getEditUrl());
  Logger.log('Responses sheet: ' + sheet.getUrl());
}
