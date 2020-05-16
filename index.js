require('@tensorflow/tfjs-node');
const tf = require('@tensorflow/tfjs');
const loadCSV = require('./load-csv');
const CSVOptions = {
  shuffle: true,
  splitTest: 10,
  dataColumns: ['lat', 'long', 'sqft_lot'],
  labelColumns: ['price'],
};

function knn(features, labels, predictionPoint, k) {
  const {mean, variance} = tf.moments(features, 0); // ): get y axis
  const scaledPrediction = predictionPoint.sub(mean).div(variance.pow(0.5));

  return (
    features
      .sub(mean) // Standardization Step
      .div(variance.pow(0.5))
      .sub(scaledPrediction) // Distance Calculation Step
      .pow(2)
      .sum(1)
      .pow(0.5)
      .expandDims(1) // rest KNN operations Step
      .concat(labels, 1)
      .unstack()
      .sort((a, b) => (a.get(0) > b.get(0) ? 1 : -1))
      .slice(0, k)
      .reduce((acc, pair) => acc + pair.get(1), 0) / k
  );
}

let {features, labels, testLabels, testFeatures} = loadCSV('./kc_house_data.csv', CSVOptions);

// convert simple arrays to Tensors
features = tf.tensor(features);
labels = tf.tensor(labels);

testFeatures.forEach((testPoint, idx) => {
  const res = knn(features, labels, tf.tensor(testPoint), 10);
  const err = (testLabels[idx][0] - res) / testLabels[idx][0];

  console.log(`Guess: ${res}, ${testLabels[idx][0]} | Error:  ${Number(err * 100).toFixed(2)}%`);
});
