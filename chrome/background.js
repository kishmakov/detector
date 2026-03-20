import { SERVER_URL } from './config.js';

chrome.runtime.onInstalled.addListener(() => {
  chrome.contextMenus.create({
    id: 'analyze',
    title: 'Analyze: "%s"',
    contexts: ['selection'],
  });
});

chrome.contextMenus.onClicked.addListener(async (info) => {
  if (info.menuItemId !== 'analyze') return;
  const resp = await fetch(`${SERVER_URL}/analyze`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ text: info.selectionText }),
  });
  const data = await resp.json();
  console.log('[detector] word_count:', data.word_count);
  chrome.notifications.create({
    type: 'basic',
    iconUrl: 'found.png',
    title: 'Detector',
    message: `Word count: ${data.word_count}`,
  });
});
