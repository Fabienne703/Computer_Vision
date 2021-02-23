#include <stdio.h>
#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <iostream>
#include <string>


#define NBRE_IMAGES 20
#define PATH_TO_PEAU_IMAGES "base/peau/"
#define PATH_TO_NON_PEAU_IMAGES "base/non-peau/"

using namespace std;
using namespace cv;


float** histogramme(string type, int echelle, float &nbre_pixels) {

	float facteur_de_reduction = (float) echelle / 256;

	//Choix du chemin d'acces d'images
	char* CHEMIN;

	if (type.compare("peau") == 0) {
		CHEMIN = PATH_TO_PEAU_IMAGES;
	} else if (type.compare("non_peau") == 0) {
		CHEMIN = PATH_TO_NON_PEAU_IMAGES;
	} else {
		cout << "Erreur de chemin d'acces! Le mauvais type de peau saisi est incorrect";
	}

	//Création de la matrice qui va contenir l'histogramme
	float ** histogramme;
	histogramme = new float*[echelle];
	for (int i = 0; i < echelle; i++) {
		histogramme[i] = new float[echelle];
		for (int j = 0; j < echelle; j++) {
			histogramme[i][j] = 0;
		}
	}

	//Construction de l'histogramme
	for (int i = 1; i <= NBRE_IMAGES; i++) {

		//définition du nom de l'image
		char nom_image[40] = "";
		strcat(nom_image, CHEMIN);

		char num[2] = "";
		sprintf(num, "%d", i);
		strcat(nom_image, num);
		strcat(nom_image, ".jpg");
        cout<<nom_image<<endl;
		//chargement de l'image
		Mat image;
		image = imread(nom_image, 1);

		if (!image.data) {
			cout << "Mauvais image" << endl;
//			exit(0);
		} else {
			// Conversion d'image RGB dans l'espace lab
			Mat resultat;
			cvtColor(image, resultat, COLOR_BGR2Lab);

			// Parcours de l'image pour remplissage de l'histogramme
			for (int k = 0; k < resultat.rows; k++) {
				for (int l = 0; l < resultat.cols; l++) {

					// choix des valeurs a et b
					int a = resultat.at<Vec3b>(k, l).val[1]
							* facteur_de_reduction;
					int b = resultat.at<Vec3b>(k, l).val[2]
							* facteur_de_reduction;

					// mise à jour des valeurs de l'histogramme
					if (image.at<Vec3b>(k, l) != Vec3b(0, 0, 0)) {

						histogramme[a][b] = histogramme[a][b] + 1;
					}
				}
			}


		}
	}

	// Lissage de l'histogramme pour améliorer la detection:
			//moyenne de la valeur des 8 pixels voisin + la valeur du pixel

			for (int i = 1; i < (echelle - 1); i++) {
				for (int j = 1; j < (echelle - 1); j++) {
					histogramme[i][j] = histogramme[i][j]
							+ (histogramme[i - 1][j - 1] + histogramme[i - 1][j]
									+ histogramme[i - 1][j + 1] + histogramme[i][j - 1]
									+ histogramme[i][j + 1] + histogramme[i + 1][j - 1]
									+ histogramme[i + 1][j] + histogramme[i + 1][j + 1])
									/ 8;
				}
			}

	//Normalisation de l'histogramme
	for (int m = 0; m < echelle; m++) {
		for (int n = 0; n < echelle; n++) {
			if(histogramme[m][n] !=0)
				nbre_pixels += histogramme[m][n];

		}
	}

	for (int m = 0; m < echelle; m++) {
			for (int n = 0; n < echelle; n++) {
				if(histogramme[m][n] !=0)
				histogramme[m][n] /= nbre_pixels;

			}
		}

	return histogramme;
}

//Evaluation des perforances du programme

void evaluation(Mat image_reference, Mat image_detectee) {

	int nbre_pixels_peau_vrai = 0;
	int nbre_pixels_peau_faux_pos = 0;
	int nbre_pixels_peau_image_reference = 0;
	int nbre_pixels_peau_faux_neg = 0;
	float performance;

	for (int i = 0; i < image_detectee.rows; i++) {
		for (int j = 0; j < image_detectee.cols; j++) {

			Vec3b Resultat = image_detectee.at<Vec3b>(i, j);
			Vec3b original = image_reference.at<Vec3b>(i, j);
			// Nombre de pixel peau correctement détecté dans le résultat
			// le pixel de l'image de résultat et de l'image de référence sont tous différent de noir
			if (Resultat != Vec3b(0, 0, 0) && original != Vec3b(0, 0, 0)) {

				nbre_pixels_peau_vrai++;
			}
			// Nombre de pixel peau mal détecté
			if (Resultat != Vec3b(0, 0, 0) && original == Vec3b(0, 0, 0)) {
				nbre_pixels_peau_faux_pos++;
			}
			// Nombre de pixel peau dans l'image de référence
			if (original != Vec3b(0, 0, 0)) {
				nbre_pixels_peau_image_reference++;
			}
		}
	}

	nbre_pixels_peau_faux_neg = nbre_pixels_peau_image_reference -nbre_pixels_peau_vrai;
	if(nbre_pixels_peau_faux_neg < 0.0)
		nbre_pixels_peau_faux_neg=0.0;

//Calcul de la performance du programme
	performance = (float)nbre_pixels_peau_vrai/(nbre_pixels_peau_vrai
					+nbre_pixels_peau_faux_pos + nbre_pixels_peau_faux_neg);
	cout << "reference :"<<nbre_pixels_peau_image_reference<< endl;
	cout << "correct :"<<nbre_pixels_peau_vrai<< endl;
	cout << "faux_positif :"<<nbre_pixels_peau_faux_pos<< endl;
	cout << "faux_negatif :"<<nbre_pixels_peau_faux_neg<< endl;

	cout << "Perfomance du programme = " << performance * 100 << " %" << endl;

}

// Détection de la peau méthode simple
Mat detection_peau_simple(float** histo_peau, float** histo_non_peau,
		Mat image_test, int echelle) {

	float facteur_de_reduction = (float) echelle / 256;
	//conversion de l'image test dans l'espace lab
	Mat resultat;
	cvtColor(image_test, resultat, COLOR_BGR2Lab);

	Mat masque(image_test.rows, image_test.cols, CV_8UC1);
	masque = Scalar(0);
	Mat sortie;
	image_test.copyTo(sortie);
	for (int k = 0; k < resultat.rows; k++) {
		for (int l = 0; l < resultat.cols; l++) {

			// choix des valeurs a et b
			int a = resultat.at<Vec3b>(k, l).val[1] * facteur_de_reduction;
			int b = resultat.at<Vec3b>(k, l).val[2] * facteur_de_reduction;

			//if(a!=0 || b!=0)
			//cout<< "a :"<< a << " b :"<< b << endl;
			// mise à jour des valeurs de l'histogramme
			if (histo_peau[a][b] < histo_non_peau[a][b]) {

				sortie.at<Vec3b>(k, l) = Vec3b(0, 0, 0);

			} else {
				masque.at<uchar>(k, l) = 255;
			}
		}
	}

	imshow("image entree", image_test);

	imshow("masque", masque);
	imshow("sortie", sortie);

	return sortie;
}

// Détection de peau par calcul de probabilité
Mat detection_peau_bayes(float** histo_peau, float** histo_non_peau,
		Mat image_test, int echelle, float seuil, float nbre_pixels_peau,
		float nbre_pixels_non_peau) {

	float facteur_de_reduction = (float) echelle / 256;
	float proba_peau = 0.0;
	float proba_non_peau = 0.0;

	//calcul des probabilités peau et non peau

	proba_peau = nbre_pixels_peau / (nbre_pixels_peau + nbre_pixels_non_peau);
	proba_non_peau = nbre_pixels_non_peau / (nbre_pixels_peau + nbre_pixels_non_peau);
//	cout << "nb peau :" << nbre_pixels_peau << endl;
//	cout << "nb non peau :" << nbre_pixels_non_peau << endl;
//	cout << "proba_peau :" << proba_peau << endl;
//	cout << "proba_non_peau :" << proba_non_peau << endl;

	//conversion de l'image test dans l'espace lab
	Mat resultat;
	cvtColor(image_test, resultat, COLOR_BGR2Lab);

	// création du masque
	Mat masque(image_test.rows, image_test.cols, CV_8UC1);
	masque = Scalar(0);

	//création de l'image résultat
	Mat sortie;
	image_test.copyTo(sortie);

	for (int k = 0; k < resultat.rows; k++) {
		for (int l = 0; l < resultat.cols; l++) {

			// choix des valeurs a et b
			int a =0, b=0;
			 a = resultat.at<Vec3b>(k, l).val[1] * facteur_de_reduction;
			 b = resultat.at<Vec3b>(k, l).val[2] * facteur_de_reduction;
			 //calcul de la probabilité de décision
			 float proba_decision = 0.0;
			 proba_decision = (histo_peau[a][b] * proba_peau)
					/ (histo_peau[a][b] * proba_peau
							+ histo_non_peau[a][b] * proba_non_peau);

			// mise à jour des valeurs de l'histogramme
			if (proba_decision > seuil) {
				masque.at<uchar>(k, l) = 255;

			} else {
				sortie.at<Vec3b>(k, l) = Vec3b(0, 0, 0);
			}
		}

	}

	//Post traitement
	int erosion_size = 1;
	int dilatation_size = 3;

	Mat dilate_element = getStructuringElement(MORPH_CROSS,
			Size(2 * dilatation_size + 1, 2 * dilatation_size + 1),
			Point(dilatation_size, dilatation_size));

	Mat erode_element = getStructuringElement(MORPH_CROSS,
			Size(2 * erosion_size + 1, 2 * erosion_size + 1),
			Point(erosion_size, erosion_size));
	dilate(sortie, sortie, dilate_element);

	erode(sortie, sortie, erode_element);

	imshow("image entree", image_test);

	imshow("masque", masque);
	imshow("sortie", sortie);


	return sortie;

}

// Affichage de l'histogramme
void histogramme_print(float ** histogramme, int echelle, string type) {

	Mat big_histogramme(256, 256, CV_8UC1);
	float valeur_maximale = 0.0;

	//Détermination de la valeur maximale de l'histogramme

	for (int i = 0; i < echelle; i++) {
		for (int j = 0; j < echelle; j++) {
			if (histogramme[i][j] > valeur_maximale)
				valeur_maximale = histogramme[i][j];
		}
	}

	//Agrandissement, normalisation de la matrice de l'histogramme et transformation en image

	for (int i = 0; i < echelle; i++) {
		for (int j = 0; j < echelle; j++) {
			for (int k = 0; k < 256/echelle; k++) {
				for (int l = 0; l < 256/echelle; l++)
					big_histogramme.at<uchar>(i * 256/echelle + k, j * 256/echelle + l) =
							saturate_cast<uchar>(
									((histogramme[i][j]) / valeur_maximale)
											* 255);
						}
		}
	}

	// Enregistrement de l'histogramme
	char nom_histogramme[50] = "";
	strcat(nom_histogramme, "histogramme/histogramme_");
	if (type.compare("peau") == 0) {
		strcat(nom_histogramme, "peau(a,b)");
	} else {
		strcat(nom_histogramme, "non_peau(a,b)");
	}
	strcat(nom_histogramme, ".jpg");
	if (!imwrite(nom_histogramme, big_histogramme))
		cout << "Erreur lors de l'enregistrement" << endl;

	// Affichage de l'histogramme
	imshow(nom_histogramme, big_histogramme);
}

// Fonction principale
int main(int argc, char** argv) {

	int echelle = 0;
	float seuil = 0.0;
	echelle = atoi(argv[1]);
	seuil = atof(argv[2]);
	float ** histo_peau = NULL;
	float ** histo_non_peau = NULL;
	float nbre_pixels_peau = 0;
	float nbre_pixels_non_peau = 0;
	char* arg_nom = argv[3];
	char nom_image_test[50]= "";
	strcat(nom_image_test,"base/test/");
	strcat(nom_image_test,arg_nom);

	// Lecture de l'image entrée
	Mat image_entre;
	image_entre = imread(nom_image_test, 1);

	char nom_image_reference[30] = PATH_TO_PEAU_IMAGES;
	strcat(nom_image_reference,arg_nom);

	// Lecture de l'image de référence
	Mat image_reference;
	image_reference = imread(nom_image_reference, 1);
	imshow("image reference dans la base", image_reference);


	Mat image_detectee;

	// calcul des histogrammes
	histo_peau = histogramme("peau", echelle, nbre_pixels_peau);

	histo_non_peau = histogramme("non_peau", echelle, nbre_pixels_non_peau);


	image_detectee = detection_peau_bayes(histo_peau, histo_non_peau,
			image_entre, echelle, seuil, nbre_pixels_peau, nbre_pixels_non_peau);

	char nom_image_resultat[30] ="";
		strcat(nom_image_resultat,"resultat/");
		strcat(nom_image_resultat,"resultat_image_");
		strcat(nom_image_resultat,arg_nom);
		if (!imwrite(nom_image_resultat, image_detectee))
				cout << "Erreur lors de l'enregistrement" << endl;

	evaluation(image_reference, image_detectee);
	histogramme_print(histo_peau,echelle,"peau");
	histogramme_print(histo_non_peau,echelle,"non_peau");
	waitKey(0);
	return 0;
}
