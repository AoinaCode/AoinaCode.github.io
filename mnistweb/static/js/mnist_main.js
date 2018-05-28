let DrawPint = TwoDinZeroArr();
let model = null;
let ModelName="None";
const ModelLabel=document.getElementsByName("NowModel");
let IsTraing = false;
let IsDrawing = false;
let VistualGrayscale = false;
let CanvasDrawColor = null;
let CanvasDrawSize = null;

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
		CanvasDrawColor = document.getElementsByName("CanvasPaintColor")[0].value
		CanvasDrawSize = parseInt(document.getElementsByName("CanvasPaintSizeRange")[0].value)
	}
	function drawing(even){
		if(IsDrawing==true){
			const rect = canvas.getBoundingClientRect();
			const x = even.clientX - parseInt(rect.left);
			const y = even.clientY - parseInt(rect.top);
			var ctx = canvas.getContext('2d');
			ctx.fillStyle=CanvasDrawColor;
			ctx.beginPath();
			ctx.arc(x,y,CanvasDrawSize,0,Math.PI*2,true)
			ctx.closePath();
			ctx.fill();
			const drawx = parseInt(x/10);
			const drawy = parseInt(y/10);
			//console.log("x:",drawx,",y:",drawy);
			if(VistualGrayscale===true){
				if(drawx<27 && drawy<27){
					DrawPint[drawy+1][drawx] = 0.5;
					DrawPint[drawy+1][drawx-1] = 0.5;
					DrawPint[drawy+1][drawx+1] = 0.5;
					DrawPint[drawy][drawx] =1;
					DrawPint[drawy][drawx-1] = 0.5;
					DrawPint[drawy][drawx+1] = 0.5;
					DrawPint[drawy-1][drawx] = 0.5;
					DrawPint[drawy-1][drawx-1] = 0.5;
					DrawPint[drawy-1][drawx+1] = 0.5;
				}else{
					IsDrawing=false;
					console.log("超出邊界");
				}
			}else if(VistualGrayscale===false){
				if(drawx<28 && drawy<28){
					DrawPint[drawy][drawx] =1;
				}else{
					IsDrawing=false;
					console.log("超出邊界");
				}
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

function ChangeCanvasSet(action){
	if(action==="color"){
		CanvasDrawColor = document.getElementsByName("CanvasPaintColor")[0].value
	}else if (action==="size"){
		CanvasDrawSize = parseInt(document.getElementsByName("CanvasPaintSizeRange")[0].value)
		const Label = document.getElementsByName("CanvasPaintSizeLabel")[0]
		Label.innerText = "Value:"+CanvasDrawSize
	}else if (action=="VistualGrayscale"){
		const Obj = document.getElementsByName("VistualGrayscale")[0]
		const Label = document.getElementsByName("VistualGrayscaleLabel")[0]
		if(Obj.checked ===true){
			Obj.checked=true
			VistualGrayscale=true
			Label.innerText="開啟"
			
		}else{
			Obj.checked=false
			VistualGrayscale=false
			Label.innerText="關閉"
			
		}
	}
}

function GetCanvasNum(){
	const canvas= document.getElementById('TestMnistDrawCanvas')
	const context = canvas.getContext('2d')
	const ValNum = document.getElementsByName('ShowCanvasNum')[0]
	if(model!=null){
		const CanvasPoint = tf.tensor2d(DrawPint).reshape([1,28,28,1])
		//console.log(DrawPint)
		const prediction = model.predict(CanvasPoint)
		const num =  prediction.argMax(1).dataSync()[0]
		ValNum.innerHTML="數字:"+num
		//console.log(num)
		context.clearRect(0, 0, canvas.width, canvas.height)
		DrawPint=TwoDinZeroArr()
	}else{
		context.clearRect(0, 0, canvas.width, canvas.height)
		DrawPint=TwoDinZeroArr()
	}
}

function  TwoDinZeroArr(){
	const array = [];
	for(let i=0;i<28;i++){
		const n = new Array(28).fill(0);
		array.push(n)
	}
	/*
	const array = [];
	for (let i = 0; i < dimensions[0]; ++i) {
		array.push(dimensions.length == 1 ? 0 : TwoDinZeroArr(dimensions.slice(1)));
	}
	*/
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
	ShowCanvas();
	const DownloadModelBtn =document.getElementsByName("DownloadModel")[0];
	const lossValueslb =document.getElementsByName("Val_loss")[0];
	const accuracyValueslb = document.getElementsByName("Val_acc")[0];
	const trainingbachlb = document.getElementsByName("TraingBach")[0];
	DownloadModelBtn.className="collapse";
	lossValueslb.innerText="---";
	accuracyValueslb.innerText="---";
	trainingbachlb.innerText="---";
}

//---

function SetTrainModel(){
	const conv1_Act = document.getElementsByName("Conv_1Activaion")[0].value;
	const conv2_Act = document.getElementsByName("Conv_2Activaion")[0].value
	model = tf.sequential();
	model.add(tf.layers.conv2d({inputShape: [28, 28, 1],kernelSize: 5,filters: 8,strides: 1,activation: conv1_Act,kernelInitializer: 'varianceScaling'}));
	model.add(tf.layers.maxPooling2d({poolSize: [2, 2], strides: [2, 2]}));
	model.add(tf.layers.conv2d({kernelSize: 5,filters: 16,strides: 1,
	activation: conv2_Act,kernelInitializer: 'varianceScaling'}));
	model.add(tf.layers.maxPooling2d({poolSize: [2, 2], strides: [2, 2]}));
	model.add(tf.layers.flatten());
	model.add(tf.layers.dense({units: 10, kernelInitializer: 'varianceScaling', activation: 'softmax'}));
	const LEARNING_RATE = 0.15;
	const optimizer = tf.train.sgd(LEARNING_RATE);
	model.compile({optimizer: optimizer,loss: 'categoricalCrossentropy',metrics: ['accuracy']});
	console.log("MNIST訓練參數讀取完成");
}

function TrainModel(){
	if(IsTraing==false){
		if(model==null){
			SetTrainModel();
			StartTrainModel();
		}else{
			$('#TrainingOldModel').modal();
		}
	}
}


async function StartTrainModel(){
	IsTraing=true;
	const Training_btn = document.getElementsByName("Training")[0];
	const DownloadModelBtn =document.getElementsByName("DownloadModel")[0];
	DownloadModelBtn.className="collapse";
	Training_btn.innerHTML="MNIST資料集加載中...";
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
		if (i % TEST_ITERATION_FREQUENCY === 0) {
			testBatch = TrainData.nextTestBatch(TEST_BATCH_SIZE);
			validationData = [testBatch.xs.reshape([TEST_BATCH_SIZE, 28, 28, 1]), testBatch.labels];
		}
		const history = await model.fit(batch.xs.reshape([BATCH_SIZE, 28, 28, 1]), batch.labels,{batchSize: BATCH_SIZE, validationData, epochs: 1});
		const loss = history.history.loss[0];
		const accuracy = history.history.acc[0];
		if (testBatch != null) {
			trainingbachlb.innerText = parseInt(i)+10;
			accuracyValueslb.innerText =  (accuracy*100).toFixed(2)+"%";
			lossValueslb.innerText=loss.toFixed(2);
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

