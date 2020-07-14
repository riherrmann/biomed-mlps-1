package integration

import de.huberlin.biomed.main
import org.junit.Test
import java.io.ByteArrayInputStream
import java.io.ByteArrayOutputStream
import java.io.PrintStream
import kotlin.test.assertTrue

class Integration
{
	@Test
	fun `it processes given documents`()
	{
		val Docs = "Maternal variables associated with physiologic stress and perinatal complications in preterm infants. Complications of prematurity may be related to dysregulation of the hypothalamic-pituitary-adrenal axis in preterm infants. Increased intrauterine exposure to cortisol may be responsible for adverse prenatal programming and subsequent dysfunction of the infant's hypothalamic-pituitary-adrenal axis. The aim of the study was to describe maternal social variables and their association with infant cortisol levels and complications of prematurity. Preterm infants <32 weeks' gestation were recruited. Primary outcomes were development of complications of prematurity and physiologic stress response, represented by cord blood and salivary cortisol levels on first day of life. Descriptive statistics and comparative analyses were performed. Fifteen of 31 infants enrolled developed a complication of prematurity. Infants of greater gestational age when prenatal care was established had lower cord blood cortisol (p = 0.009) and trended a higher risk of necrotizing enterocolitis (p = 0.069). Infants whose mothers smoked more showed significantly different salivary cortisol distributions on day 1 (p = 0.037), and were at greater risk for intraventricular hemorrhage (p = 0.018). The association between maternal social variables, hypothalamic-pituitary-adrenal axis dysregulation, and complications of prematurity supports the research model of physiologic dysregulation/allostatic load as a mechanism for complications in preterm infants. More research is warranted to investigate associations between maternal social variables, maternal stress levels, and adverse prenatal programming of the infant hypothalamic-pituitary-adrenal axis.\n" +
			"Severity of Household Food Insecurity Is Positively Associated with Mental Disorders among Children and Adolescents in the United States. Household food insecurity and mental disorders are both prevalent conditions among children and adolescents (i.e., youth) in the United States. Although some research has examined the association between the 2 conditions, it is not known whether more severe food insecurity is differently associated with mental disorders in youth. We investigated the association between severity of household food insecurity and mental disorders among children (aged 4-11 y) and adolescents (aged 12-17 y) using valid and reliable measures of both household food security status and mental disorders. We analyzed cross-sectional data on 16,918 children and 14,143 adolescents whose families participated in the 2011-2014 National Health Interview Survey. The brief Strengths and Difficulties Questionnaire and the 10-item USDA Household Food Security Survey Module were used to measure mental disorders and food security status, respectively. Multinomial logistic regressions were used to test the association between household food security status and mental disorders in youth. There was a significant linear trend in ORs, such that as severity of household food insecurity increased so did the odds of youth having a mental disorder (P < 0.001). Other selected results included the following: compared with food-secure households, youth in marginally food-secure households had higher odds of having a mental disorder with impairment [child OR: 1.26 (95% CI: 1.05, 1.52); adolescent OR: 1.33 (95% CI: 1.05, 1.68)]. In addition, compared with food-secure households, youth in very-low-food-secure households had higher odds of having a mental disorder with severe impairment [child OR: 2.55 (95% CI: 1.90, 3.43); adolescent OR: 3.44 (95% CI: 2.50, 4.75)]. The severity of household food insecurity is positively associated with mental disorders among both children and adolescents in the United States. These results suggest that improving household food security status has the potential to reduce mental disorders among US youth.\n" +
			"Mediation of episodic memory performance by the executive function network in patients with amnestic mild cognitive impairment: a resting-state functional MRI study. Deficits in episodic memory (EM) are a hallmark clinical symptom of patients with amnestic mild cognitive impairment (aMCI). Impairments in executive function (EF) are widely considered to exacerbate memory deficits and to increase the risk of conversion from aMCI to Alzheimer's disease (AD). However, the specific mechanisms underlying the interaction between executive dysfunction and memory deficits in aMCI patients remain unclear. Thus, the present study utilized resting-state functional magnetic resonance imaging (fMRI) scans of the EF network and the EM network to investigate this relationship in 79 aMCI patients and 119 healthy controls (HC). The seeds were obtained from the results of a regional homogeneity (ReHo) analysis. Functional connectivity (FC) within the EM network was determined using a seed in the right retrosplenial cortex (RSC), and FC within EF network was assessed using seeds in the right dorsolateral prefrontal cortex (DLPFC). There was a significant negative correlation between EM scores and EF scores in both the aMCI and HC groups. Compared to the HC group, aMCI patients had reduced right RSC connectivity but enhanced right DLPFC connectivity. The overlapping brain regions between the EM and EF networks were associated with FC in the right inferior parietal lobule (IPL) in the right RSC network, and in the bilateral middle cingulate cortex (MCC) and left IPL in the right DLPFC network. A mediation analysis revealed that the EF network had an indirect positive effect on EM performance in the aMCI patients. The present findings provide new insights into the neural mechanisms underlying the interaction between impaired EF and memory deficits in aMCI patients and suggest that the EF network may mediate EM performance in this population."

		val Stdin = System.`in`
		val Stdout = System.out
		val Input = ByteArrayInputStream( Docs.toByteArray() )
		val Output = ByteArrayOutputStream()

		System.setOut( PrintStream( Output ) )
		System.setIn( Input )

		main( arrayOf( "-f", "n" ) )

		assertTrue( Output.toString().trim().isNotEmpty() )

		System.setOut( Stdout )
		System.setIn( Stdin )

	}
}