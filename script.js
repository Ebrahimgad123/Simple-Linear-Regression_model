const fs = require('fs');
const regression = require('regression');

// Read data from the CSV file
const csvData = fs.readFileSync('./Salary Data.csv', 'utf8');

// Parse the CSV data into an array of arrays
const data = csvData
  .trim() // Remove any leading/trailing whitespace
  .split('\n') // Split by newline character
  .map(row => row.split(',')) // Split each row by comma
  .map(row => row.map(val => parseFloat(val.trim()))) // Convert strings to numbers
  .filter(row => !row.includes(NaN)); // Filter out rows with NaN values

// Perform linear regression
const result = regression.linear(data);

// Extract gradient (slope) and y-intercept from the result
const gradient = result.equation[0];
const yIntercept = result.equation[1];

// Predict salary for 3.2 years of experience
const yearsOfExperience = 4;
const predictedSalary = gradient * yearsOfExperience + yIntercept;

// Output the results
console.log('Gradient:', gradient);
console.log('Y-Intercept:', yIntercept);
console.log('Predicted Salary for 4 years of experience:', predictedSalary);
