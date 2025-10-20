# comp3000

## Project Title
Automating feline pain detection using facial analysis and machine learning

## Supervisor
Lauren Ansell

## Project Vision

This Project Vision was built off Paulo Caroli’s ‘Write the Product Vision’ article. (Caroli, 2022)

For anyone with a cat in their care, this project will aim to use machine learning to analyse images of cats and predict their level of acute pain.

Cats are highly adept at hiding their pain. (Horwitz and Rodan, 2018) Owners can fail to spot early signs of pain in their pet, missing their cat’s pain until it progresses enough that they cannot hide it as well anymore. (Gruen et al., 2022)

This project aims to lower the incident rate of this issue; shelters, veterinarians, and the general cat-owning public would be able to upload a photo of their cat and get an accurate reading of acute pain levels.

In the UK, between 2016 and 2023, the average vet price has increased by 63%. (Competition and Markets Authority, 2025) Catching pain earlier can prevent an acute pain scenario from becoming a chronic (or long-lasting) pain scenario (Downing and Della Rocca, 2023), meaning the pet can spend less time (and the owner less money) at the vet. This can also prevent the heartbreaking scenario of an owner having to choose between a large debt and putting their pet down. (McNamee et al., 2025)

In addition, pain can cause or worsen problematic behaviour in cats, and it is difficult to ascertain the connection between the two. (Mills et al., 2020) Problematic behaviour is one of the top reasons that cats are returned to shelters after being adopted. (Mundschau and Suchak, 2023). It is possible that if a shelter could use a pain-detecting model to check on their cats more regularly than they can get vets to check on them, they may be able to spot more painful cats, making sure to heal their condition before adopting them out. That way, the number of cats being returned to a shelter could decrease, freeing up space for other cats in need of assistance.

To understand this project, it is important to learn about the Feline Grimace Scale (FGS). It is a scale that is used to assess pain in cats by analysing small changes in their facial expression. There are 5 ‘Action Units’ (AUs) used in the FGS – ear position, orbital tightening (in other words, narrowing of the eyes), whisker change, the tension of the muzzle, and head position. (Cayetano Evangelista, 2018) These different features have a pain score corresponding to each one depending on their state, and added together, you get the pain score.

Below are 2 Minimum Viable Product (MVP) ideas. They cover the likely risk of a painful cat database not being obtained by Sprint 1.

If a suitable painful cat dataset cannot be obtained (Plan A):
1)	Use a normal cat database.
2)	Select Action Units (AUs) to prioritise depending on the size of the database. If it is on the smaller side (perhaps 1000 images or less), prioritise the eyes, muzzle ratio and ear angle. This is as Feline Grimace Scale (FGS) scorers had the most agreement with each other using these features. (Cayetano Evangelista, 2018) Add whiskers, head positions, and other FGS features later (post MVP). The idea behind this is that facial feature mapping can be very time-consuming and it may be unrealistic to complete the entire face by December.
3)	Calculate and use ratios to predict painful or pain-free classification. (Cayetano Evangelista, 2018) This takes away the reliance of needing a painful database.
4)	Later, if we get a painful cat dataset, we can use it to test the model.
If a suitable painful cat dataset can be obtained (Plan B):
1)	Either do the same as above but with earlier testing against the painful cat database, OR:
2)	Use something like a heat map CNN to highlight the areas flagged as ‘important’ to the CNN.
3)	See if the areas flagged up by the heat map are the same areas flagged by the Feline Grimace Scale, give a predicted classification (pain-free or painful, and if possible what level on the Feline Grimace Scale), and then show the user the relevant FGS guidance to see if their analysis agrees.
a.	The idea is, if the heat map lights up areas it sees as relevant to pain in cats (like a tightened muzzle), it will flag up what has resulted in the pain score and the user can check.

If a proof of concept of the idea can be created by March, it would be ideal to create a web app that users could test this on. 

Realistically, this project will be a small piece of the puzzle as it is a deep and complex problem to tackle.

## References
Caroli, P. (2022) Lean Inception, martinfowler.com. Available at: https://martinfowler.com/articles/lean-inception/ (Accessed: 18 October 2025).

Evangelista, M. (2018) ‘Facial expressions of pain in cats: development of the Feline Grimace Scale’, in ResearchGate. Available at: https://www.researchgate.net/publication/323830301_Facial_expressions_of_pain_in_cats_development_of_the_Feline_Grimace_Scale (Accessed: 13 October 2025).

Competition and Markets Authority (2025) Major reforms would require vet businesses to make fundamental changes to the way they support pet owners, GOV.UK. Available at: https://www.gov.uk/government/news/major-reforms-would-require-vet-businesses-to-make-fundamental-changes-to-the-way-they-support-pet-owners (Accessed: 15 October 2025).

Downing, R. and Della Rocca, G. (2023) ‘Pain in Pets: Beyond Physiology’, Animals : an Open Access Journal from MDPI, 13(3), p. 355. Available at: https://doi.org/10.3390/ani13030355.

Gruen, M.E. et al. (2022) ‘2022 AAHA Pain Management Guidelines for Dogs and Cats’, Journal of the American Animal Hospital Association, 58(2), pp. 55–76. Available at: https://doi.org/10.5326/JAAHA-MS-7292.

Horwitz, D.F. and Rodan, I. (2018) ‘Behavioral awareness in the feline consultation: Understanding physical and emotional health’, Journal of Feline Medicine and Surgery, 20(5), pp. 423–436. Available at: https://doi.org/10.1177/1098612X18771204.

McNamee, B.S. et al. (2025) Vets should be made to publish prices, competition watchdog says, BBC News. Available at: https://www.bbc.co.uk/news/articles/c201r14z6r3o (Accessed: 15 October 2025).

Mills, D.S. et al. (2020) ‘Pain and Problem Behavior in Cats and Dogs’, Animals, 10(2), p. 318. Available at: https://doi.org/10.3390/ani10020318.

Mundschau, V. and Suchak, M. (2023) ‘When and Why Cats Are Returned to Shelters’, Animals : an Open Access Journal from MDPI, 13(2), p. 243. Available at: https://doi.org/10.3390/ani13020243.

University of Plymouth (2025) Risk Assessment Form (RA1). University of Plymouth. Available at: https://www.psy.plymouth.ac.uk/home/Documents/RA-LinkLabs.pdf (Accessed: 13 October 2025).
