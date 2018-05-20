

let DrawPint = TwoDinZeroArr([28,28])
let model = null
let ModelName="None";
const IMAGE_SIZE = 784;
const NUM_CLASSES = 10;
const NUM_DATASET_ELEMENTS = 65000;
const NUM_TRAIN_ELEMENTS = 55000;
const NUM_TEST_ELEMENTS = NUM_DATASET_ELEMENTS - NUM_TRAIN_ELEMENTS;
const MNIST_IMAGES_SPRITE_PATH ='https://storage.googleapis.com/learnjs-data/model-builder/mnist_images.png';
const MNIST_LABELS_PATH ='https://storage.googleapis.com/learnjs-data/model-builder/mnist_labels_uint8';
const ModelLabel=document.getElementsByName("NowModel");
let IsTraing = false;
let IsDrawing = false;

function SetConvas(){
		const canvas= document.getElementById('TestMnistDrawCanvas');
		if (canvas.getContext){
				canvas.addEventListener('mousedown',function(e){
					IsDrawing=true
				} )
				canvas.addEventListener('mouseup',function(e){
					IsDrawing=false
				})
				canvas.addEventListener('mousemove',drawing);
		}
		function drawing(even){
			if(IsDrawing==true){
				const rect = canvas.getBoundingClientRect();
				const x = even.clientX - parseInt(rect.left);
				const y = even.clientY - parseInt(rect.top);
				var ctx = canvas.getContext('2d');
				ctx.beginPath();
				ctx.arc(x,y,10,0,Math.PI*2,true)
				ctx.closePath();
				ctx.fill();
				const drawx = parseInt(x/10);
				const drawy = parseInt(y/10);
				//console.log("x:",drawx,",y:",drawy);
				if(drawx<28 && drawy<28){
					DrawPint[drawy][drawx] = 1;
				}else{
					IsDrawing=false;
					console.log("超出邊界");
				}
				
			}
		}
	
}

function ShowCanvas(){
	const MNISTCanvasBtn = document.getElementsByName("MNISTCanvasBtn")[0];
	const MNISTCanvasObj = document.getElementsByName("MNISTCanvasObj")[0];
	if(model!=null){
		MNISTCanvasBtn.innerHTML="測試模型辨識";
		if(MNISTCanvasObj.className==="collapse"){
			MNISTCanvasObj.className="collapse show";
		}else{
			MNISTCanvasObj.className="collapse";
		}
	}else{
		MNISTCanvasBtn.innerHTML="沒有讀取到模型,請重試";
		MNISTCanvasObj.className="collapse";
	}
}

function GetCanvasNum(){
	const canvas= document.getElementById('TestMnistDrawCanvas')
	const context = canvas.getContext('2d')
	const ValNum = document.getElementsByName('ShowCanvasNum')[0]
	if(model!=null){
		const CanvasPoint = tf.tensor2d(DrawPint).reshape([1,28,28,1])
		const prediction = model.predict(CanvasPoint)
		const num =  prediction.argMax(1).dataSync()[0]
		ValNum.innerHTML="數字:"+num
		console.log(num)
		context.clearRect(0, 0, canvas.width, canvas.height)
		DrawPint=TwoDinZeroArr([28,28])
	}else{
		context.clearRect(0, 0, canvas.width, canvas.height)
		DrawPint=TwoDinZeroArr([28,28])
	}
}

function  TwoDinZeroArr(dimensions){
	const array = [];
	for (let i = 0; i < dimensions[0]; ++i) {
		array.push(dimensions.length == 1 ? 0 : TwoDinZeroArr(dimensions.slice(1)));
	}
	return array;
}

async function UploadFile(){
	
	const UploadData_Model = document.getElementById('json-upload');
	const UploadData_Weight = document.getElementById('weights-upload');
	model = await tf.loadModel(tf.io.browserFiles([UploadData_Model.files[0], UploadData_Weight.files[0]]));
	ModelName=UploadData_Model.files[0].name;
	ModelLabel[0].innerHTML=ModelName;
	
}

function Download(){
	console.log("下載");
	model.save('downloads://'+ModelName);
}

function ModelOptions(action){
	const UploadModel = document.getElementById('UploadModelSet');
	const TrainModel = document.getElementById('TrainModelSet');
	if(action=="UploadModel"){
		if(UploadModel.className=="collapse show"){
			UploadModel.className="collapse";
		}else{
			UploadModel.className="collapse show";
		}
		TrainModel.className="collapse";
	}else if(action=="TrainModel"){
		if(TrainModel.className=="collapse show"){
			TrainModel.className="collapse";
		}else{
			TrainModel.className="collapse show";
		}
		UploadModel.className="collapse";
		
	}
}

function ChangeScrllBarLabel(object_name,label_name){
	const object=document.getElementsByName(object_name)[0]
	const label = document.getElementsByName(label_name)[0]
	label.innerHTML=object.value
}

function DeletModel(){
	model = null;
	ModelName="--";
	ModelLabel[0].innerHTML=ModelName;
}

//---

function SetTrainModel(){
	model = tf.sequential();
	model.add(tf.layers.conv2d({
	inputShape: [28, 28, 1],
	kernelSize: 5,
	filters: 8,
	strides: 1,
	activation: 'relu',
	kernelInitializer: 'varianceScaling'
	}));
	model.add(tf.layers.maxPooling2d({poolSize: [2, 2], strides: [2, 2]}));
	model.add(tf.layers.conv2d({
	kernelSize: 5,
	filters: 16,
	strides: 1,
	activation: 'relu',
	kernelInitializer: 'varianceScaling'
	}));
	model.add(tf.layers.maxPooling2d({poolSize: [2, 2], strides: [2, 2]}));
	model.add(tf.layers.flatten());
	model.add(tf.layers.dense(
		{units: 10, kernelInitializer: 'varianceScaling', activation: 'softmax'}));
	const LEARNING_RATE = 0.15;
	const optimizer = tf.train.sgd(LEARNING_RATE);
	model.compile({
	optimizer: optimizer,
	loss: 'categoricalCrossentropy',
	metrics: ['accuracy'],
	});
	console.log("MNIST訓練參數讀取完成");
}

async function TrainModel(){
	if(IsTraing==false){
		IsTraing=true;
		const Training_btn = document.getElementsByName("Training")[0];
		const DownloadModelBtn =document.getElementsByName("DownloadModel")[0];
		DownloadModelBtn.className="collapse";
		Training_btn.innerHTML="MNIST資料集加載中...";
		SetTrainModel();
		ModelName = document.getElementsByName("ModelName")[0].value;
		//const BATCH_SIZE = 64;
		const BATCH_SIZE = parseInt(document.getElementsByName("TrainBachSizeRange")[0].value);
		//const TRAIN_BATCHES = 300;
		const TRAIN_BATCHES = parseInt(document.getElementsByName("TrainBachRange")[0].value);
		const TEST_BATCH_SIZE = 1000;
		
		const TEST_ITERATION_FREQUENCY = 10;
		const TrainData = new MnistData();
		await TrainData.load();
		Training_btn.innerHTML="加載中完成";
		console.log("MNIST資料集讀取完成");


		
		
		const lossValueslb =document.getElementsByName("Val_loss")[0];
		const accuracyValueslb = document.getElementsByName("Val_acc")[0];
		const trainingbachlb = document.getElementsByName("TraingBach")[0];
		Training_btn.innerHTML="模型訓練中...";
		for (let i = 0; i < TRAIN_BATCHES; i++) {
			const batch = TrainData.nextTrainBatch(BATCH_SIZE);

			let testBatch;
			let validationData;
			// Every few batches test the accuracy of the mode.
			if (i % TEST_ITERATION_FREQUENCY === 0) {
			testBatch = TrainData.nextTestBatch(TEST_BATCH_SIZE);
			validationData = [
				testBatch.xs.reshape([TEST_BATCH_SIZE, 28, 28, 1]), testBatch.labels
			];
			}

			// The entire dataset doesn't fit into memory so we call fit repeatedly
			// with batches.
			const history = await model.fit(
				batch.xs.reshape([BATCH_SIZE, 28, 28, 1]), batch.labels,
				{batchSize: BATCH_SIZE, validationData, epochs: 1});

			const loss = history.history.loss[0];
			const accuracy = history.history.acc[0];
			if (testBatch != null) {
				trainingbachlb.innerHTML = parseInt(i)+10;
				accuracyValueslb.innerHTML =  (accuracy*100).toFixed(2)+"%";
				lossValueslb.innerHTML=loss.toFixed(2);
			}

			batch.xs.dispose();
			batch.labels.dispose();
			if (testBatch != null) {
			testBatch.xs.dispose();
			testBatch.labels.dispose();
			}
			
			await tf.nextFrame();
		}
		Training_btn.innerHTML="Training";
		IsTraing=false;
		ModelLabel[0].innerHTML=ModelName;
		DownloadModelBtn.className="collapse show";
	}
	
	
}

class MnistData {
	
	constructor() {
		this.shuffledTrainIndex = 0;
	  	this.shuffledTestIndex = 0;
	  	
		
	}
	async load() {

		// Make a request for the MNIST sprited image.
		const img = new Image();
		const canvas = document.createElement('canvas');
		const ctx = canvas.getContext('2d');
		const imgRequest = new Promise((resolve, reject) => {
			img.crossOrigin = '';
			img.onload = () => {
			img.width = img.naturalWidth;
			img.height = img.naturalHeight;
	
			const datasetBytesBuffer =
				new ArrayBuffer(NUM_DATASET_ELEMENTS * IMAGE_SIZE * 4);
	
			const chunkSize = 5000;
			canvas.width = img.width;
			canvas.height = chunkSize;
	
			for (let i = 0; i < NUM_DATASET_ELEMENTS / chunkSize; i++) {
				const datasetBytesView = new Float32Array(
					datasetBytesBuffer, i * IMAGE_SIZE * chunkSize * 4,
					IMAGE_SIZE * chunkSize);
				ctx.drawImage(
					img, 0, i * chunkSize, img.width, chunkSize, 0, 0, img.width,
					chunkSize);
	
				const imageData = ctx.getImageData(0, 0, canvas.width, canvas.height);
	
				for (let j = 0; j < imageData.data.length / 4; j++) {
				// All channels hold an equal value since the image is grayscale, so
				// just read the red channel.
				datasetBytesView[j] = imageData.data[j * 4] / 255;
				}
			}
			this.datasetImages = new Float32Array(datasetBytesBuffer);
	
			resolve();
			};
			img.src = MNIST_IMAGES_SPRITE_PATH;
		});
	
		const labelsRequest = fetch(MNIST_LABELS_PATH);
		const [imgResponse, labelsResponse] =
			await Promise.all([imgRequest, labelsRequest]);
	
		this.datasetLabels = new Uint8Array(await labelsResponse.arrayBuffer());
	
		// Create shuffled indices into the train/test set for when we select a
		// random dataset element for training / validation.
		this.trainIndices = tf.util.createShuffledIndices(NUM_TRAIN_ELEMENTS);
		this.testIndices = tf.util.createShuffledIndices(NUM_TEST_ELEMENTS);
	
		// Slice the the images and labels into train and test sets.
		this.trainImages =
			this.datasetImages.slice(0, IMAGE_SIZE * NUM_TRAIN_ELEMENTS);
		this.testImages = this.datasetImages.slice(IMAGE_SIZE * NUM_TRAIN_ELEMENTS);
		this.trainLabels =
			this.datasetLabels.slice(0, NUM_CLASSES * NUM_TRAIN_ELEMENTS);
		this.testLabels =
			this.datasetLabels.slice(NUM_CLASSES * NUM_TRAIN_ELEMENTS);
	}
  
	nextTrainBatch(batchSize) {
	  return this.nextBatch(
		  batchSize, [this.trainImages, this.trainLabels], () => {
			this.shuffledTrainIndex =
				(this.shuffledTrainIndex + 1) % this.trainIndices.length;
			return this.trainIndices[this.shuffledTrainIndex];
		  });
	}
  
	nextTestBatch(batchSize) {
	  return this.nextBatch(batchSize, [this.testImages, this.testLabels], () => {
		this.shuffledTestIndex =
			(this.shuffledTestIndex + 1) % this.testIndices.length;
		return this.testIndices[this.shuffledTestIndex];
	  });
	}
  
	nextBatch(batchSize, data, index) {
	  const batchImagesArray = new Float32Array(batchSize * IMAGE_SIZE);
	  const batchLabelsArray = new Uint8Array(batchSize * NUM_CLASSES);
  
	  for (let i = 0; i < batchSize; i++) {
		const idx = index();
  
		const image =
			data[0].slice(idx * IMAGE_SIZE, idx * IMAGE_SIZE + IMAGE_SIZE);
		batchImagesArray.set(image, i * IMAGE_SIZE);
  
		const label =
			data[1].slice(idx * NUM_CLASSES, idx * NUM_CLASSES + NUM_CLASSES);
		batchLabelsArray.set(label, i * NUM_CLASSES);
	  }
  
	  const xs = tf.tensor2d(batchImagesArray, [batchSize, IMAGE_SIZE]);
	  const labels = tf.tensor2d(batchLabelsArray, [batchSize, NUM_CLASSES]);
  
	  return {xs, labels};
	}
}