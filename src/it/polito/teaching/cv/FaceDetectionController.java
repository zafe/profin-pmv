package it.polito.teaching.cv;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.List;
import java.util.Random;
import java.util.concurrent.Executors;
import java.util.concurrent.ScheduledExecutorService;
import java.util.concurrent.TimeUnit;

import javax.xml.transform.Source;

import org.opencv.core.Core;
import org.opencv.core.Mat;
import org.opencv.core.MatOfInt;
import org.opencv.core.MatOfRect;
import org.opencv.core.Point;
import org.opencv.core.Rect;
import org.opencv.core.Scalar;
import org.opencv.core.Size;
import org.opencv.face.FaceRecognizer;
import org.opencv.face.LBPHFaceRecognizer;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.Imgproc;
import org.opencv.objdetect.CascadeClassifier;
import org.opencv.objdetect.Objdetect;
import org.opencv.videoio.VideoCapture;

import it.polito.elite.teaching.cv.utils.Utils;
import javafx.event.Event;
import javafx.fxml.FXML;
import javafx.scene.control.Button;
import javafx.scene.control.CheckBox;
import javafx.scene.control.TextField;
import javafx.scene.image.Image;
import javafx.scene.image.ImageView;

/**
 * The controller associated with the only view of our application. The
 * application logic is implemented here. It handles the button for
 * starting/stopping the camera, the acquired video stream, the relative
 * controls and the face detection/tracking.
 * 
 * @author <a href="mailto:luigi.derussis@polito.it">Luigi De Russis</a>
 * @version 1.1 (2015-11-10)
 * @since 1.0 (2014-01-10)
 * 		
 */
public class FaceDetectionController
{
	// FXML buttons
	@FXML
	private Button cameraButton;
	@FXML
	private Button fotografiarButton;
	@FXML
	private Button entrenarButton;
	@FXML
	private Button reconocerButton;
	// the FXML area for showing the current frame
	@FXML
	private ImageView originalFrame;
	// checkboxes for enabling/disabling a classifier
	@FXML
	private CheckBox haarClassifier;
	@FXML
	private CheckBox lbpClassifier;
	@FXML
	private TextField nombrePersonaTextField;
	
	// a timer for acquiring the video stream
	private ScheduledExecutorService timer;
	// the OpenCV object that performs the video capture
	private VideoCapture capture;
	// a flag to change the button behavior
	private boolean cameraActive;
	
	// face cascade classifier
	private CascadeClassifier faceCascade;
	private int absoluteFaceSize;
	
	private String nombrePersona = "persona";
	private int numeroImagen = 0;
	
	FaceRecognizer faceRecognizer = LBPHFaceRecognizer.create(); 
	int banderaEntrenada  = 0;
	List<String> namesIndexList = new ArrayList<>();


	
	/**
	 * Init the controller, at start time
	 */
	protected void init()
	{
		this.capture = new VideoCapture();
		this.faceCascade = new CascadeClassifier();
		this.absoluteFaceSize = 0;
		
		// set a fixed width for the frame
		originalFrame.setFitWidth(600);
		// preserve image ratio
		originalFrame.setPreserveRatio(true);
	}
	
	/**
	 * The action triggered by pushing the button on the GUI
	 */
	@FXML
	protected void startCamera()
	{	
		if (!this.cameraActive)
		{
			// disable setting checkboxes
			this.haarClassifier.setDisable(true);
			this.lbpClassifier.setDisable(true);
			
			// start the video capture
			this.capture.open(0);
			
			// is the video stream available?
			if (this.capture.isOpened())
			{
				this.cameraActive = true;
				// grab a frame every 33 ms (30 frames/sec)
				Runnable frameGrabber = new Runnable() {
					
					@Override
					public void run()
					{
						// effectively grab and process a single frame
						Mat frame = grabFrame();
						// convert and show the frame
						Image imageToShow = Utils.mat2Image(frame);
						
						updateImageView(originalFrame, imageToShow);
					}
				};
				
				this.timer = Executors.newSingleThreadScheduledExecutor();
				this.timer.scheduleAtFixedRate(frameGrabber, 0, 33, TimeUnit.MILLISECONDS);
				
				// update the button content
				this.cameraButton.setText("Detener Camara");
			}
			else
			{
				// log the error
				System.err.println("Failed to open the camera connection...");
			}
		}
		else
		{
			// the camera is not active at this point
			this.cameraActive = false;
			// update again the button content
			this.cameraButton.setText("Detener Camara");
			// enable classifiers checkboxes
			this.haarClassifier.setDisable(false);
			this.lbpClassifier.setDisable(false);
			
			// stop the timer
			this.stopAcquisition();
		}
	}
	
	
	//Tomar foto para entrenar
	@FXML
	protected void takePhoto(){
		
	      nombrePersona = nombrePersonaTextField.getText();
	      String file2 = "/Users/fernando/Desktop/entrenamiento/"+ nombrePersona +"-"+ numeroImagen +".jpg"; 
	      Mat matrix = grabFrame();
	      
	      MatOfRect faces = new MatOfRect();
	      Mat grayFrame = new Mat();
			
		  // convert the frame in gray scale
		  Imgproc.cvtColor(matrix, grayFrame, Imgproc.COLOR_BGR2GRAY);
		  // equalize the frame histogram to improve the result
		  Imgproc.equalizeHist(grayFrame, grayFrame);
			
		  // compute minimum face size (20% of the frame height, in our case)
		  if (this.absoluteFaceSize == 0)
			{
				int height = grayFrame.rows();
				if (Math.round(height * 0.2f) > 0)
				{
					this.absoluteFaceSize = Math.round(height * 0.2f);
				}
			}
			
			// detect faces
			this.faceCascade.detectMultiScale(grayFrame, faces, 1.1, 2, 0 | Objdetect.CASCADE_SCALE_IMAGE,
					new Size(this.absoluteFaceSize, this.absoluteFaceSize), new Size());
					
			// each rectangle in faces is a face: draw them!
			Rect[] facesArray = faces.toArray();
			
			for (int i = 0; i < facesArray.length; i++){
				
				
					Mat mat =	matrix.submat(facesArray[i]);		
				  
			      Imgcodecs.imwrite(file2, mat); 

			}
	      
	      
	      
	      System.out.println("Imagen guardada de " + nombrePersona);
	      numeroImagen++;
		
	}
	
	/**
	 * Get a frame from the opened video stream (if any)
	 * 
	 * @return the {@link Image} to show
	 */
	private Mat grabFrame()
	{
		Mat frame = new Mat();
		
		// check if the capture is open
		if (this.capture.isOpened())
		{
			try
			{
				// read the current frame
				this.capture.read(frame);
				
				// if the frame is not empty, process it
				if (!frame.empty())
				{
					// face detection
					this.detectAndDisplay(frame);
				}
				
			}
			catch (Exception e)
			{
				// log the (full) error
				System.err.println("Exception during the image elaboration: " + e);
			}
		}
		
		return frame;
	}
	
	/**
	 * Method for face detection and tracking
	 * 
	 * @param frame
	 *            it looks for faces in this frame
	 */
	private void detectAndDisplay(Mat frame)
	{
		MatOfRect faces = new MatOfRect();
		Mat grayFrame = new Mat();
		
		// convert the frame in gray scale
		Imgproc.cvtColor(frame, grayFrame, Imgproc.COLOR_BGR2GRAY);
		// equalize the frame histogram to improve the result
		Imgproc.equalizeHist(grayFrame, grayFrame);
		
		// compute minimum face size (20% of the frame height, in our case)
		if (this.absoluteFaceSize == 0)
		{
			int height = grayFrame.rows();
			if (Math.round(height * 0.2f) > 0)
			{
				this.absoluteFaceSize = Math.round(height * 0.2f);
			}
		}
		
		// detect faces
		this.faceCascade.detectMultiScale(grayFrame, faces, 1.1, 2, 0 | Objdetect.CASCADE_SCALE_IMAGE,
				new Size(this.absoluteFaceSize, this.absoluteFaceSize), new Size());
				
		// each rectangle in faces is a face: draw them!
		Rect[] facesArray = faces.toArray();
		Point centerPoint;
		
		for (int i = 0; i < facesArray.length; i++){
			
			/*
			double x_Axis = facesArray[i].tl().x + ((facesArray[i].br().x - facesArray[i].tl().x)/2);
		    double y_Axis = facesArray[i].tl().y + (facesArray[i].height/2);
		    
		    centerPoint = new Point(x_Axis, y_Axis);
		   
		    System.out.println("\n tl x: " + facesArray[i].tl().x + " y: " + facesArray[i].tl().y);
		    System.out.println(" br x: " + facesArray[i].br().x + " y: " + facesArray[i].br().y);
		    System.out.println(" x: " + centerPoint.x + " y: " + centerPoint.y);


			Imgproc.circle(frame, centerPoint, (int) (( facesArray[i].br().x - facesArray[i].tl().x)/2), new Scalar(0,255,0),3);
			*/
			Imgproc.rectangle(frame, facesArray[i].tl(), facesArray[i].br(), new Scalar(0, 255, 0), 3);
			
			if(nombrePersonaTextField.getText() != ""){
				
				Imgproc.putText(frame, 
						nombrePersonaTextField.getText(), 							//text
						facesArray[i].tl(), 				//origin point
						Core.FONT_HERSHEY_TRIPLEX,   //fontFace
						2, 									//fontScale
						new Scalar(0,255,0)); 				//color
				
			}
			
			
			//reconoce los rostros
			if(banderaEntrenada == 1){
					Mat mat =	frame.submat(facesArray[i]);		
					Imgproc.cvtColor(mat, mat, Imgproc.COLOR_BGR2GRAY); // Convert image to grayscale or it won't work
					String nombre = namesIndexList.get(faceRecognizer.predict_label(mat));
					System.out.println("Rostro reconocido: " + nombre);
					
					Imgproc.putText(frame, 
							nombre, 							//text
							facesArray[i].tl(), 				//origin point
							Core.FONT_HERSHEY_COMPLEX,   //fontFace
							4, 									//fontScale
							new Scalar(255,255,0)); 
		}
			
			
		}
		
		
		}
	
	@FXML
	protected void trainRecognizer(){
		
		
		Path path = Paths.get("/Users/fernando/Desktop/entrenamiento/");
		ArrayList<Mat> sourceImages = new ArrayList<>();
		List<Integer> namesIntList = new ArrayList<>();
		try {
			Files.list(path).forEach(file -> {
			    String filename = file.getFileName().toString();
			    if (filename.contains("-") && filename.endsWith("jpg")){
			        String personName = filename.substring(0, filename.indexOf("-")); // e.g. edd from edd-1.jpg
			        if (!namesIndexList.contains(personName)){
			            namesIndexList.add(personName);
			        }
			        Mat image = Imgcodecs.imread(file.toString());
			        Imgproc.cvtColor(image, image, Imgproc.COLOR_BGR2GRAY); // Convert image to grayscale or it won't work with the face learner
			        sourceImages.add(image);
			        namesIntList.add(namesIndexList.indexOf(personName));
			    }
			});
		} catch (IOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
		MatOfInt matOfInt = new MatOfInt();
		matOfInt.fromList(namesIntList);
		faceRecognizer.train(sourceImages, matOfInt);
		
		banderaEntrenada = 1;
		
	}
	
	
	
	/**
	 * The action triggered by selecting the Haar Classifier checkbox. It loads
	 * the trained set to be used for frontal face detection.
	 */
	@FXML
	protected void haarSelected(Event event)
	{
		// check whether the lpb checkbox is selected and deselect it
		if (this.lbpClassifier.isSelected())
			this.lbpClassifier.setSelected(false);
			
		this.checkboxSelection("resources/haarcascades/haarcascade_frontalface_alt.xml");

	}
	
	/**
	 * The action triggered by selecting the LBP Classifier checkbox. It loads
	 * the trained set to be used for frontal face detection.
	 */
	@FXML
	protected void lbpSelected(Event event)
	{
		// check whether the haar checkbox is selected and deselect it
		if (this.haarClassifier.isSelected())
			this.haarClassifier.setSelected(false);
			
		this.checkboxSelection("resources/lbpcascades/lbpcascade_frontalface.xml");
	}
	
	/**
	 * Method for loading a classifier trained set from disk
	 * 
	 * @param classifierPath
	 *            the path on disk where a classifier trained set is located
	 */
	private void checkboxSelection(String classifierPath)
	{
		// load the classifier(s)
		this.faceCascade.load(classifierPath);
		
		// now the video capture can start
		this.cameraButton.setDisable(false);
	}
	
	/**
	 * Stop the acquisition from the camera and release all the resources
	 */
	private void stopAcquisition()
	{
		if (this.timer!=null && !this.timer.isShutdown())
		{
			try
			{
				// stop the timer
				this.timer.shutdown();
				this.timer.awaitTermination(33, TimeUnit.MILLISECONDS);
			}
			catch (InterruptedException e)
			{
				// log any exception
				System.err.println("Exception in stopping the frame capture, trying to release the camera now... " + e);
			}
		}
		
		if (this.capture.isOpened())
		{
			// release the camera
			this.capture.release();
		}
	}
	
	/**
	 * Update the {@link ImageView} in the JavaFX main thread
	 * 
	 * @param view
	 *            the {@link ImageView} to update
	 * @param image
	 *            the {@link Image} to show
	 */
	private void updateImageView(ImageView view, Image image)
	{
		Utils.onFXThread(view.imageProperty(), image);
	}
	
	/**
	 * On application close, stop the acquisition from the camera
	 */
	protected void setClosed()
	{
		this.stopAcquisition();
	}
	
}
