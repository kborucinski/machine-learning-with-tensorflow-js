const dataUrl =
  "https://raw.githubusercontent.com/kborucinski/machine-learning-with-tensorflow-js/master/houses.json";

const getData = async url => {
  const result = await fetch(url);
  const data = await result.json();

  // 1 ft = 0.3048 m
  return data.map(({ grLivArea, salePrice }) => ({
    area: Math.pow(Math.sqrt(grLivArea) * 0.3048, 2),
    salePrice
  }));
};

const createModel = () => {
  const model = tf.sequential();

  model.add(tf.layers.dense({ inputShape: [1], units: 1 }));
  model.add(tf.layers.dense({ units: 1 }));

  return model;
};

const convertToTensor = data =>
  tf.tidy(() => {
    tf.util.shuffle(data);

    const inputs = data.map(({ area }) => area);
    const labels = data.map(({ salePrice }) => salePrice);

    const inputTensor = tf.tensor2d(inputs, [inputs.length, 1]);
    const labelTensor = tf.tensor2d(labels, [labels.length, 1]);

    const inputMax = inputTensor.max();
    const inputMin = inputTensor.min();

    const labelMax = labelTensor.max();
    const labelMin = labelTensor.min();

    const normalizedInputs = inputTensor
      .sub(inputMin)
      .div(inputMax.sub(inputMin));
    const normalizedLabels = labelTensor
      .sub(labelMin)
      .div(labelMax.sub(labelMin));

    return {
      inputs: normalizedInputs,
      labels: normalizedLabels,
      inputMax,
      inputMin,
      labelMax,
      labelMin
    };
  });

const trainModel = async (model, inputs, labels) => {
  model.compile({
    optimizer: tf.train.adam(),
    loss: tf.losses.meanSquaredError,
    metrics: ["mse"]
  });

  return await model.fit(inputs, labels, {
    batchSize: 40,
    epochs: 100,
    shuffle: true,
    callbacks: tfvis.show.fitCallbacks(
      { name: "Przebieg treningu" },
      ["loss", "mse"],
      { callbacks: ["onEpochEnd"], height: 300 }
    )
  });
};

const testModel = (model, inputData, normalizationData) => {
  const { inputMax, inputMin, labelMin, labelMax } = normalizationData;

  const [xs, predictions] = tf.tidy(() => {
    const xs = tf.linspace(0, 1, 70);
    const predictions = model.predict(xs.reshape([70, 1]));

    const unnormalizedInputs = xs.mul(inputMax.sub(inputMin)).add(inputMin);
    const unnormalizedPredictions = predictions
      .mul(labelMax.sub(labelMin))
      .add(labelMin);

    return [unnormalizedInputs.dataSync(), unnormalizedPredictions.dataSync()];
  });

  const predictedPoints = Array.from(xs).map((x, index) => ({
    x,
    y: predictions[index]
  }));

  const originalPoints = inputData.map(({ area, salePrice }) => ({
    x: area,
    y: salePrice
  }));

  tfvis.render.scatterplot(
    { name: "Dane prognozowane a Dane oryginalne" },
    {
      values: [originalPoints, predictedPoints],
      series: ["oryginalne", "prognozowane"]
    },
    {
      height: 300,
      xLabel: "Powierzchnia domu",
      yLabel: "Cena",
      zoomToFit: true
    }
  );
};

const run = async () => {
  const data = await getData(dataUrl);

  const values = data.map(({ area, salePrice }) => ({
    x: area,
    y: salePrice
  }));

  tfvis.render.scatterplot(
    { name: "Powierzchnia domu a Cena domu" },
    { values },
    {
      height: 300,
      xLabel: "Powierzchnia domu",
      yLabel: "Cena domu",
      zoomToFit: true
    }
  );

  const model = createModel();
  tfvis.show.modelSummary({ name: "Utworzone warstwy" }, model);

  const tensor = convertToTensor(data);
  const { inputs, labels } = tensor;

  await trainModel(model, inputs, labels);

  testModel(model, data, tensor);
};

document.addEventListener("DOMContentLoaded", run);
