require('@tensorflow/tfjs-node');
const tf = require('@tensorflow/tfjs');
const loadCSV = require('./load-csv');
const CSVOptions = {
  shuffle: true,
  splitTest: 10,
  dataColumns: ['lat', 'long'],
  labelColumns: ['price'],
};

function knn(features, labels, predictionPoint, k) {
  return (
    features
      .sub(predictionPoint)
      .pow(2)
      .sum(1)
      .pow(0.5)
      .expandDims(1)
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

const res = knn(features, labels, tf.tensor(testFeatures[0]), 10);
console.log('Guess: ', res, testLabels[0][0]);
