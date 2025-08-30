import { render, screen } from '@testing-library/react';
import App from './App';


test('renders app bar title', () => {
  render(<App />);
  const titleElement = screen.getByText(/Application: Tomato Disease Classification/i);
  expect(titleElement).toBeInTheDocument();
});



// test('renders learn react link', () => {
//   render(<App />);
//   const linkElement = screen.getByText(/learn react/i);
//   expect(linkElement).toBeInTheDocument();
// });

