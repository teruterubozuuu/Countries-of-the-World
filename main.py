import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


# Load data from CSV file
asia = pd.read_csv('asia.csv')

#-------------ASIA---------------#

#Asia x
population_asia = asia['population'].values
literacy_asia = asia['literacy'].values
climate_asia = asia['climate'].values
birthrate_asia = asia['birthrate'].values
deathrate_asia = asia['deathrate'].values

#Asia y
gdp_asia = asia['gdp'].values


#Population & GDP
coeffs_population_gdp_asia = np.polyfit(population_asia, gdp_asia, 1)
line_population_gdp_asia = np.poly1d(coeffs_population_gdp_asia)

corr_coef_pop_gdp_asia = np.corrcoef(population_asia, gdp_asia)[0, 1]

plt.subplot(2, 1, 1)
plt.scatter(population_asia, gdp_asia)
plt.plot(population_asia, line_population_gdp_asia(population_asia), color='red')

plt.xlabel('Population')
plt.ylabel('GDP')
plt.title('Asia')
plt.text(0.5, 0.6, f"y = {line_population_gdp_asia} \nCorrelation Coefficient = {corr_coef_pop_gdp_asia:.2f}", transform=plt.gca().transAxes)


#Literacy & GDP
coeffs_literacy_gdp_asia = np.polyfit(literacy_asia, gdp_asia, 1)
line_literacy_gdp_asia = np.poly1d(coeffs_literacy_gdp_asia)
plt.subplot(2,1,2)
plt.scatter(literacy_asia,gdp_asia)
plt.plot(literacy_asia,line_literacy_gdp_asia(literacy_asia),color='red')

corr_coef_literacy_gdp_asia = np.corrcoef(literacy_asia, gdp_asia)[0, 1]

plt.xlabel('Literacy')
plt.ylabel('GDP')
plt.text(0.4, 0.6, f"y = {line_literacy_gdp_asia} \nCorrelation coefficient = {corr_coef_literacy_gdp_asia:.2f}", transform=plt.gca().transAxes)
plt.subplots_adjust(hspace=0.5)
plt.show(block=False)

plt.figure()


#Climate and GDP
coeffs_climate_gdp_asia = np.polyfit(climate_asia, gdp_asia, 1)
line_climate_gdp_asia = np.poly1d(coeffs_climate_gdp_asia)

corr_coef_climate_gdp_asia = np.corrcoef(climate_asia, gdp_asia)[0, 1]

plt.subplot(2,1,1)
plt.scatter(climate_asia, gdp_asia)
plt.plot(climate_asia, line_climate_gdp_asia(climate_asia), color='red')

plt.xlabel('Climate')
plt.ylabel('GDP')
plt.title('Asia')
plt.text(0.5, 0.6, f"y = {line_climate_gdp_asia} \nCorrelation Coefficient = {corr_coef_climate_gdp_asia:.2f}", transform=plt.gca().transAxes)
plt.subplots_adjust(hspace=0.5)
plt.show(block=False)



#Birthrate and GDP
coeffs_birthrate_gdp_asia = np.polyfit(birthrate_asia, gdp_asia, 1)
line_birthrate_gdp_asia = np.poly1d(coeffs_birthrate_gdp_asia)

corr_coef_birthrate_gdp_asia = np.corrcoef(birthrate_asia, gdp_asia)[0, 1]

plt.subplot(2,1,2)
plt.scatter(birthrate_asia, gdp_asia)
plt.plot(birthrate_asia, line_birthrate_gdp_asia(birthrate_asia), color='red')

plt.xlabel('Birthrate')
plt.ylabel('GDP')
plt.text(0.5, 0.6, f"y = {line_birthrate_gdp_asia} \nCorrelation Coefficient = {corr_coef_birthrate_gdp_asia:.2f}", transform=plt.gca().transAxes)
plt.show(block=False)

plt.figure()
#Deathrate and GDP
coeffs_deathrate_gdp_asia = np.polyfit(deathrate_asia, gdp_asia, 1)
line_deathrate_gdp_asia = np.poly1d(coeffs_deathrate_gdp_asia)

corr_coef_deathrate_gdp_asia = np.corrcoef(deathrate_asia, gdp_asia)[0, 1]

plt.subplot(2,1,1)
plt.scatter(deathrate_asia, gdp_asia)
plt.plot(deathrate_asia, line_deathrate_gdp_asia(deathrate_asia), color='red')

plt.xlabel('Deathrate')
plt.ylabel('GDP')
plt.title('Asia')
plt.text(0.5, 0.6, f"y = {line_deathrate_gdp_asia} \nCorrelation Coefficient = {corr_coef_deathrate_gdp_asia:.2f}", transform=plt.gca().transAxes)
plt.show(block=False)

#-------------BALTICS---------------#

baltics = pd.read_csv('baltics.csv')

#Baltics x
population_baltics = baltics['population'].values
literacy_baltics = baltics['literacy'].values
climate_baltics = baltics['climate'].values
birthrate_baltics = baltics['birthrate'].values
deathrate_baltics = baltics['deathrate'].values

#Baltics y
gdp_baltics = baltics['gdp'].values

#Population & GDP
coeffs_population_gdp_baltics = np.polyfit(population_baltics, gdp_baltics, 1)
line_population_gdp_baltics = np.poly1d(coeffs_population_gdp_baltics)

corr_coef_pop_gdp_baltics = np.corrcoef(population_baltics, gdp_baltics)[0, 1]

plt.figure()
plt.subplot(2, 1, 1)
plt.scatter(population_baltics, gdp_baltics)
plt.plot(population_baltics, line_population_gdp_baltics(population_baltics), color='red')

plt.xlabel('Population')
plt.ylabel('GDP')
plt.title('Baltics')
plt.text(0.5, 0.6, f"y = {line_population_gdp_baltics} \nCorrelation Coefficient = {corr_coef_pop_gdp_baltics:.2f}", transform=plt.gca().transAxes)


#Literacy & GDP
coeffs_literacy_gdp_baltics = np.polyfit(literacy_baltics, gdp_baltics, 1)
line_literacy_gdp_baltics = np.poly1d(coeffs_literacy_gdp_baltics)
plt.subplot(2,1,2)
plt.scatter(literacy_baltics,gdp_baltics)
plt.plot(literacy_baltics,line_literacy_gdp_baltics(literacy_baltics),color='red')

corr_coef_literacy_gdp_baltics = np.corrcoef(literacy_baltics, gdp_baltics)[0, 1]

plt.xlabel('Literacy')
plt.ylabel('GDP')
plt.text(0.4, 0.6, f"y = {line_literacy_gdp_baltics} \nCorrelation coefficient = {corr_coef_literacy_gdp_baltics:.2f}", transform=plt.gca().transAxes)
plt.subplots_adjust(hspace=0.5)
plt.show(block=False)

plt.figure()


#Climate and GDP
coeffs_climate_gdp_baltics = np.polyfit(climate_baltics, gdp_baltics, 1)
line_climate_gdp_baltics = np.poly1d(coeffs_climate_gdp_baltics)

corr_coef_climate_gdp_baltics = np.corrcoef(climate_baltics, gdp_baltics)[0, 1]

plt.subplot(2,1,1)
plt.scatter(climate_baltics, gdp_baltics)
plt.plot(climate_baltics, line_climate_gdp_baltics(climate_baltics), color='red')

plt.xlabel('Climate')
plt.ylabel('GDP')
plt.title('Baltics')
plt.text(0.5, 0.6, f"y = {line_climate_gdp_baltics} \nCorrelation Coefficient = {corr_coef_climate_gdp_baltics:.2f}", transform=plt.gca().transAxes)
plt.subplots_adjust(hspace=0.5)
plt.show(block=False)



#Birthrate and GDP
coeffs_birthrate_gdp_baltics = np.polyfit(birthrate_baltics, gdp_baltics, 1)
line_birthrate_gdp_baltics = np.poly1d(coeffs_birthrate_gdp_baltics)

corr_coef_birthrate_gdp_baltics = np.corrcoef(birthrate_baltics, gdp_baltics)[0, 1]

plt.subplot(2,1,2)
plt.scatter(birthrate_baltics, gdp_baltics)
plt.plot(birthrate_baltics, line_birthrate_gdp_baltics(birthrate_baltics), color='red')

plt.xlabel('Birthrate')
plt.ylabel('GDP')
plt.text(0.1, 0.6, f"y = {line_birthrate_gdp_baltics} \nCorrelation Coefficient = {corr_coef_birthrate_gdp_baltics:.2f}", transform=plt.gca().transAxes)
plt.show(block=False)

plt.figure()
#Deathrate and GDP
coeffs_deathrate_gdp_baltics = np.polyfit(deathrate_baltics, gdp_baltics, 1)
line_deathrate_gdp_baltics = np.poly1d(coeffs_deathrate_gdp_baltics)

corr_coef_deathrate_gdp_baltics = np.corrcoef(deathrate_baltics, gdp_baltics)[0, 1]

plt.subplot(2,1,1)
plt.scatter(deathrate_baltics, gdp_baltics)
plt.plot(deathrate_baltics, line_deathrate_gdp_baltics(deathrate_baltics), color='red')

plt.xlabel('Deathrate')
plt.ylabel('GDP')
plt.title('Baltics')
plt.text(0.5, 0.6, f"y = {line_deathrate_gdp_asia} \nCorrelation Coefficient = {corr_coef_deathrate_gdp_asia:.2f}", transform=plt.gca().transAxes)
plt.show(block=False)

#-------------Eastern Europe---------------#

eastern_europe = pd.read_csv('eastern_europe.csv')

#Eastern Europe x
population_eastern_europe = eastern_europe['population'].values
literacy_eastern_europe = eastern_europe['literacy'].values
climate_eastern_europe = eastern_europe['climate'].values
birthrate_eastern_europe = eastern_europe['birthrate'].values
deathrate_eastern_europe = eastern_europe['deathrate'].values

#Eastern Europe y
gdp_eastern_europe = eastern_europe['gdp'].values

#Population & GDP
coeffs_population_gdp_eastern_europe = np.polyfit(population_eastern_europe, gdp_eastern_europe, 1)
line_population_gdp_eastern_europe = np.poly1d(coeffs_population_gdp_eastern_europe)

corr_coef_pop_gdp_eastern_europe = np.corrcoef(population_eastern_europe, gdp_eastern_europe)[0, 1]

plt.figure()
plt.subplot(2, 1, 1)
plt.scatter(population_eastern_europe, gdp_eastern_europe)
plt.plot(population_eastern_europe, line_population_gdp_eastern_europe(population_eastern_europe), color='red')

plt.xlabel('Population')
plt.ylabel('GDP')
plt.title('Eastern Europe')
plt.text(0.5, 0.6, f"y = {line_population_gdp_eastern_europe} \nCorrelation Coefficient = {corr_coef_pop_gdp_eastern_europe:.2f}", transform=plt.gca().transAxes)


#Literacy & GDP
coeffs_literacy_gdp_eastern_europe = np.polyfit(literacy_eastern_europe, gdp_eastern_europe, 1)
line_literacy_gdp_eastern_europe = np.poly1d(coeffs_literacy_gdp_eastern_europe)
plt.subplot(2,1,2)
plt.scatter(literacy_eastern_europe,gdp_eastern_europe)
plt.plot(literacy_eastern_europe,line_literacy_gdp_eastern_europe(literacy_eastern_europe),color='red')

corr_coef_literacy_gdp_eastern_europe = np.corrcoef(literacy_eastern_europe, gdp_eastern_europe)[0, 1]

plt.xlabel('Literacy')
plt.ylabel('GDP')
plt.text(0.4, 0.6, f"y = {line_literacy_gdp_eastern_europe} \nCorrelation coefficient = {corr_coef_literacy_gdp_eastern_europe:.2f}", transform=plt.gca().transAxes)
plt.subplots_adjust(hspace=0.5)
plt.show(block=False)

plt.figure()


#Climate and GDP
coeffs_climate_gdp_eastern_europe = np.polyfit(climate_eastern_europe, gdp_eastern_europe, 1)
line_climate_gdp_eastern_europe = np.poly1d(coeffs_climate_gdp_eastern_europe)

corr_coef_climate_gdp_eastern_europe = np.corrcoef(climate_eastern_europe, gdp_eastern_europe)[0, 1]

plt.subplot(2,1,1)
plt.scatter(climate_eastern_europe, gdp_eastern_europe)
plt.plot(climate_eastern_europe, line_climate_gdp_eastern_europe(climate_eastern_europe), color='red')

plt.xlabel('Climate')
plt.ylabel('GDP')
plt.title('Eastern Europe')
plt.text(0.2, 0.6, f"y = {line_climate_gdp_eastern_europe} \nCorrelation Coefficient = {corr_coef_climate_gdp_eastern_europe:.2f}", transform=plt.gca().transAxes)
plt.subplots_adjust(hspace=0.5)
plt.show(block=False)



#Birthrate and GDP
coeffs_birthrate_gdp_eastern_europe = np.polyfit(birthrate_eastern_europe, gdp_eastern_europe, 1)
line_birthrate_gdp_eastern_europe = np.poly1d(coeffs_birthrate_gdp_eastern_europe)

corr_coef_birthrate_gdp_eastern_europe = np.corrcoef(birthrate_eastern_europe, gdp_eastern_europe)[0, 1]

plt.subplot(2,1,2)
plt.scatter(birthrate_eastern_europe, gdp_eastern_europe)
plt.plot(birthrate_eastern_europe, line_birthrate_gdp_eastern_europe(birthrate_eastern_europe), color='red')

plt.xlabel('Birthrate')
plt.ylabel('GDP')
plt.text(0.2, 0.6, f"y = {line_birthrate_gdp_eastern_europe} \nCorrelation Coefficient = {corr_coef_birthrate_gdp_eastern_europe:.2f}", transform=plt.gca().transAxes)
plt.show(block=False)

plt.figure()
#Deathrate and GDP
coeffs_deathrate_gdp_eastern_europe = np.polyfit(deathrate_eastern_europe, gdp_eastern_europe, 1)
line_deathrate_gdp_eastern_europe = np.poly1d(coeffs_deathrate_gdp_eastern_europe)

corr_coef_deathrate_gdp_eastern_europe = np.corrcoef(deathrate_eastern_europe, gdp_eastern_europe)[0, 1]

plt.subplot(2,1,1)
plt.scatter(deathrate_eastern_europe, gdp_eastern_europe)
plt.plot(deathrate_eastern_europe, line_deathrate_gdp_eastern_europe(deathrate_eastern_europe), color='red')

plt.xlabel('Deathrate')
plt.ylabel('GDP')
plt.title('Eastern Europe')
plt.text(0.2, 0.6, f"y = {line_deathrate_gdp_eastern_europe} \nCorrelation Coefficient = {corr_coef_deathrate_gdp_eastern_europe:.2f}", transform=plt.gca().transAxes)
plt.show(block=False)

#-------------Ind States---------------#

ind_states = pd.read_csv('ind_states.csv')

#Eastern Europe x
population_ind_states = ind_states['population'].values
literacy_ind_states = ind_states['literacy'].values
climate_ind_states = ind_states['climate'].values
birthrate_ind_states = ind_states['birthrate'].values
deathrate_ind_states = ind_states['deathrate'].values

#Eastern Europe y
gdp_ind_states = ind_states['gdp'].values

#Population & GDP
coeffs_population_gdp_ind_states = np.polyfit(population_ind_states, gdp_ind_states, 1)
line_population_gdp_ind_states = np.poly1d(coeffs_population_gdp_ind_states)

corr_coef_pop_gdp_ind_states = np.corrcoef(population_ind_states, gdp_ind_states)[0, 1]

plt.figure()
plt.subplot(2, 1, 1)
plt.scatter(population_ind_states, gdp_ind_states)
plt.plot(population_ind_states, line_population_gdp_ind_states(population_ind_states), color='red')

plt.xlabel('Population')
plt.ylabel('GDP')
plt.title('Ind States')
plt.text(0.5, 0.3, f"y = {line_population_gdp_ind_states} \nCorrelation Coefficient = {corr_coef_pop_gdp_ind_states:.2f}", transform=plt.gca().transAxes)


#Literacy & GDP
coeffs_literacy_gdp_ind_states = np.polyfit(literacy_ind_states, gdp_ind_states, 1)
line_literacy_gdp_ind_states = np.poly1d(coeffs_literacy_gdp_ind_states)
plt.subplot(2,1,2)
plt.scatter(literacy_ind_states,gdp_ind_states)
plt.plot(literacy_ind_states,line_literacy_gdp_ind_states(literacy_ind_states),color='red')

corr_coef_literacy_gdp_ind_states = np.corrcoef(literacy_ind_states, gdp_ind_states)[0, 1]

plt.xlabel('Literacy')
plt.ylabel('GDP')
plt.text(0.1, 0.6, f"y = {line_literacy_gdp_ind_states} \nCorrelation coefficient = {corr_coef_literacy_gdp_ind_states:.2f}", transform=plt.gca().transAxes)
plt.subplots_adjust(hspace=0.5)
plt.show(block=False)

plt.figure()


#Climate and GDP
coeffs_climate_gdp_ind_states = np.polyfit(climate_ind_states, gdp_ind_states, 1)
line_climate_gdp_ind_states = np.poly1d(coeffs_climate_gdp_ind_states)

corr_coef_climate_gdp_ind_states = np.corrcoef(climate_ind_states, gdp_ind_states)[0, 1]

plt.subplot(2,1,1)
plt.scatter(climate_ind_states, gdp_ind_states)
plt.plot(climate_ind_states, line_climate_gdp_ind_states(climate_ind_states), color='red')

plt.xlabel('Climate')
plt.ylabel('GDP')
plt.title('Ind States')
plt.text(0.3, 0.6, f"y = {line_climate_gdp_ind_states} \nCorrelation Coefficient = {corr_coef_climate_gdp_ind_states:.2f}", transform=plt.gca().transAxes)
plt.subplots_adjust(hspace=0.5)
plt.show(block=False)



#Birthrate and GDP
coeffs_birthrate_gdp_ind_states = np.polyfit(birthrate_ind_states, gdp_ind_states, 1)
line_birthrate_gdp_ind_states = np.poly1d(coeffs_birthrate_gdp_ind_states)

corr_coef_birthrate_gdp_ind_states = np.corrcoef(birthrate_ind_states, gdp_ind_states)[0, 1]

plt.subplot(2,1,2)
plt.scatter(birthrate_ind_states, gdp_ind_states)
plt.plot(birthrate_ind_states, line_birthrate_gdp_ind_states(birthrate_ind_states), color='red')

plt.xlabel('Birthrate')
plt.ylabel('GDP')
plt.text(0.2, 0.6, f"y = {line_birthrate_gdp_ind_states} \nCorrelation Coefficient = {corr_coef_birthrate_gdp_ind_states:.2f}", transform=plt.gca().transAxes)
plt.show(block=False)

plt.figure()
#Deathrate and GDP
coeffs_deathrate_gdp_ind_states = np.polyfit(deathrate_ind_states, gdp_ind_states, 1)
line_deathrate_gdp_ind_states = np.poly1d(coeffs_deathrate_gdp_ind_states)

corr_coef_deathrate_gdp_ind_states = np.corrcoef(deathrate_ind_states, gdp_ind_states)[0, 1]

plt.subplot(2,1,1)
plt.scatter(deathrate_ind_states, gdp_ind_states)
plt.plot(deathrate_ind_states, line_deathrate_gdp_ind_states(deathrate_ind_states), color='red')

plt.xlabel('Deathrate')
plt.ylabel('GDP')
plt.title('Ind States')
plt.text(0.1, 0.6, f"y = {line_deathrate_gdp_ind_states} \nCorrelation Coefficient = {corr_coef_deathrate_gdp_ind_states:.2f}", transform=plt.gca().transAxes)
plt.show(block=False)

#-------------Latin Amercia---------------#

latin_america = pd.read_csv('latin_america.csv')

#Latin America x
population_latin_america = latin_america['population'].values
literacy_latin_america = latin_america['literacy'].values
climate_latin_america = latin_america['climate'].values
birthrate_latin_america = latin_america['birthrate'].values
deathrate_latin_america = latin_america['deathrate'].values

#Latin America y
gdp_latin_america = latin_america['gdp'].values

#Population & GDP
coeffs_population_gdp_latin_america = np.polyfit(population_latin_america, gdp_latin_america, 1)
line_population_gdp_latin_america = np.poly1d(coeffs_population_gdp_latin_america)

corr_coef_pop_gdp_latin_america = np.corrcoef(population_latin_america, gdp_latin_america)[0, 1]

plt.figure()
plt.subplot(2, 1, 1)
plt.scatter(population_latin_america, gdp_latin_america)
plt.plot(population_latin_america, line_population_gdp_latin_america(population_latin_america), color='red')

plt.xlabel('Population')
plt.ylabel('GDP')
plt.title('Latin America')
plt.text(0.5, 0.5, f"y = {line_population_gdp_latin_america} \nCorrelation Coefficient = {corr_coef_pop_gdp_latin_america:.2f}", transform=plt.gca().transAxes)


#Literacy & GDP
coeffs_literacy_gdp_latin_america = np.polyfit(literacy_latin_america, gdp_latin_america, 1)
line_literacy_gdp_latin_america= np.poly1d(coeffs_literacy_gdp_latin_america)
plt.subplot(2,1,2)
plt.scatter(literacy_latin_america,gdp_latin_america)
plt.plot(literacy_latin_america,line_literacy_gdp_latin_america(literacy_latin_america),color='red')

corr_coef_literacy_gdp_latin_america = np.corrcoef(literacy_latin_america, gdp_latin_america)[0, 1]

plt.xlabel('Literacy')
plt.ylabel('GDP')
plt.text(0.1, 0.6, f"y = {line_literacy_gdp_latin_america} \nCorrelation coefficient = {corr_coef_literacy_gdp_latin_america:.2f}", transform=plt.gca().transAxes)
plt.subplots_adjust(hspace=0.5)
plt.show(block=False)

plt.figure()


#Climate and GDP
coeffs_climate_gdp_latin_america = np.polyfit(climate_latin_america, gdp_latin_america, 1)
line_climate_gdp_latin_america = np.poly1d(coeffs_climate_gdp_latin_america)

corr_coef_climate_gdp_latin_america = np.corrcoef(climate_latin_america, gdp_latin_america)[0, 1]

plt.subplot(2,1,1)
plt.scatter(climate_latin_america, gdp_latin_america)
plt.plot(climate_latin_america, line_climate_gdp_latin_america(climate_latin_america), color='red')

plt.xlabel('Climate')
plt.ylabel('GDP')
plt.title('Latin America')
plt.text(0.5, 0.6, f"y = {line_climate_gdp_latin_america} \nCorrelation Coefficient = {corr_coef_climate_gdp_latin_america:.2f}", transform=plt.gca().transAxes)
plt.subplots_adjust(hspace=0.5)
plt.show(block=False)



#Birthrate and GDP
coeffs_birthrate_gdp_latin_america = np.polyfit(birthrate_latin_america, gdp_latin_america, 1)
line_birthrate_gdp_latin_america = np.poly1d(coeffs_birthrate_gdp_latin_america)

corr_coef_birthrate_gdp_latin_america = np.corrcoef(birthrate_latin_america, gdp_latin_america)[0, 1]

plt.subplot(2,1,2)
plt.scatter(birthrate_latin_america, gdp_latin_america)
plt.plot(birthrate_latin_america, line_birthrate_gdp_latin_america(birthrate_latin_america), color='red')

plt.xlabel('Birthrate')
plt.ylabel('GDP')
plt.text(0.5, 0.6, f"y = {line_birthrate_gdp_latin_america} \nCorrelation Coefficient = {corr_coef_birthrate_gdp_latin_america:.2f}", transform=plt.gca().transAxes)
plt.show(block=False)

plt.figure()

#Deathrate and GDP
coeffs_deathrate_gdp_latin_america = np.polyfit(deathrate_latin_america, gdp_latin_america, 1)
line_deathrate_gdp_latin_america = np.poly1d(coeffs_deathrate_gdp_latin_america)

corr_coef_deathrate_gdp_latin_america = np.corrcoef(deathrate_latin_america, gdp_latin_america)[0, 1]

plt.subplot(2,1,1)
plt.scatter(deathrate_latin_america, gdp_latin_america)
plt.plot(deathrate_latin_america, line_deathrate_gdp_latin_america(deathrate_latin_america), color='red')

plt.xlabel('Deathrate')
plt.ylabel('GDP')
plt.title('Latin America')
plt.text(0.1, 0.6, f"y = {line_deathrate_gdp_latin_america} \nCorrelation Coefficient = {corr_coef_deathrate_gdp_latin_america:.2f}", transform=plt.gca().transAxes)
plt.show(block=False)

#-------------Mid East---------------#

mideast = pd.read_csv('mideast.csv')

#Mid East x
population_mideast = mideast['population'].values
literacy_mideast = mideast['literacy'].values
climate_mideast =mideast['climate'].values
birthrate_mideast = mideast['birthrate'].values
deathrate_mideast = mideast['deathrate'].values

#Mid East y
gdp_mideast = mideast['gdp'].values

#Population & GDP
coeffs_population_gdp_mideast = np.polyfit(population_mideast, gdp_mideast, 1)
line_population_gdp_mideast = np.poly1d(coeffs_population_gdp_mideast)

corr_coef_pop_gdp_mideast = np.corrcoef(population_mideast, gdp_mideast)[0, 1]

plt.figure()
plt.subplot(2, 1, 1)
plt.scatter(population_mideast, gdp_mideast)
plt.plot(population_mideast, line_population_gdp_mideast(population_mideast), color='red')

plt.xlabel('Population')
plt.ylabel('GDP')
plt.title('Mid East')
plt.text(0.5, 0.5, f"y = {line_population_gdp_mideast} \nCorrelation Coefficient = {corr_coef_pop_gdp_mideast:.2f}", transform=plt.gca().transAxes)


#Literacy & GDP
coeffs_literacy_gdp_mideast = np.polyfit(literacy_mideast, gdp_mideast, 1)
line_literacy_gdp_mideast= np.poly1d(coeffs_literacy_gdp_mideast)
plt.subplot(2,1,2)
plt.scatter(literacy_mideast,gdp_mideast)
plt.plot(literacy_mideast,line_literacy_gdp_mideast(literacy_mideast),color='red')

corr_coef_literacy_gdp_mideast = np.corrcoef(literacy_mideast, gdp_mideast)[0, 1]

plt.xlabel('Literacy')
plt.ylabel('GDP')
plt.text(0.1, 0.6, f"y = {line_literacy_gdp_mideast} \nCorrelation coefficient = {corr_coef_literacy_gdp_mideast:.2f}", transform=plt.gca().transAxes)
plt.subplots_adjust(hspace=0.5)
plt.show(block=False)

plt.figure()


#Climate and GDP
coeffs_climate_gdp_mideast = np.polyfit(climate_mideast, gdp_mideast, 1)
line_climate_gdp_mideast = np.poly1d(coeffs_climate_gdp_mideast)

corr_coef_climate_gdp_mideast = np.corrcoef(climate_mideast, gdp_mideast)[0, 1]

plt.subplot(2,1,1)
plt.scatter(climate_mideast, gdp_mideast)
plt.plot(climate_mideast, line_climate_gdp_mideast(climate_mideast), color='red')

plt.xlabel('Climate')
plt.ylabel('GDP')
plt.title('Mid East')
plt.text(0.5, 0.6, f"y = {line_climate_gdp_mideast} \nCorrelation Coefficient = {corr_coef_climate_gdp_mideast:.2f}", transform=plt.gca().transAxes)
plt.subplots_adjust(hspace=0.5)
plt.show(block=False)



#Birthrate and GDP
coeffs_birthrate_gdp_mideast = np.polyfit(birthrate_mideast, gdp_mideast, 1)
line_birthrate_gdp_mideast = np.poly1d(coeffs_birthrate_gdp_mideast)

corr_coef_birthrate_gdp_mideast = np.corrcoef(birthrate_mideast, gdp_mideast)[0, 1]

plt.subplot(2,1,2)
plt.scatter(birthrate_mideast, gdp_mideast)
plt.plot(birthrate_mideast, line_birthrate_gdp_mideast(birthrate_mideast), color='red')

plt.xlabel('Birthrate')
plt.ylabel('GDP')
plt.text(0.5, 0.6, f"y = {line_birthrate_gdp_mideast} \nCorrelation Coefficient = {corr_coef_birthrate_gdp_mideast:.2f}", transform=plt.gca().transAxes)
plt.show(block=False)

plt.figure()
#Deathrate and GDP
coeffs_deathrate_gdp_mideast = np.polyfit(deathrate_mideast, gdp_mideast, 1)
line_deathrate_gdp_mideast = np.poly1d(coeffs_deathrate_gdp_mideast)

corr_coef_deathrate_gdp_mideast = np.corrcoef(deathrate_mideast, gdp_mideast)[0, 1]

plt.subplot(2,1,1)
plt.scatter(deathrate_mideast, gdp_mideast)
plt.plot(deathrate_mideast, line_deathrate_gdp_mideast(deathrate_mideast), color='red')

plt.xlabel('Deathrate')
plt.ylabel('GDP')
plt.title('Mid East')
plt.text(0.1, 0.6, f"y = {line_deathrate_gdp_mideast} \nCorrelation Coefficient = {corr_coef_deathrate_gdp_mideast:.2f}", transform=plt.gca().transAxes)
plt.show(block=False)


#-------------North Africa---------------#

north_africa = pd.read_csv('north_africa.csv')

#North Africa x
population_north_africa = north_africa['population'].values
literacy_north_africa = north_africa['literacy'].values
climate_north_africa =north_africa['climate'].values
birthrate_north_africa = north_africa['birthrate'].values
deathrate_north_africa = north_africa['deathrate'].values

#North Africa y
gdp_north_africa = north_africa['gdp'].values

#Population & GDP
coeffs_population_gdp_north_africa = np.polyfit(population_north_africa, gdp_north_africa, 1)
line_population_gdp_north_africa = np.poly1d(coeffs_population_gdp_north_africa)

corr_coef_pop_gdp_north_africa = np.corrcoef(population_north_africa, gdp_north_africa)[0, 1]

plt.figure()
plt.subplot(2, 1, 1)
plt.scatter(population_north_africa, gdp_north_africa)
plt.plot(population_north_africa, line_population_gdp_north_africa(population_north_africa), color='red')

plt.xlabel('Population')
plt.ylabel('GDP')
plt.title('North Africa')
plt.text(0.5, 0.2, f"y = {line_population_gdp_north_africa} \nCorrelation Coefficient = {corr_coef_pop_gdp_north_africa:.2f}", transform=plt.gca().transAxes)


#Literacy & GDP
coeffs_literacy_gdp_north_africa = np.polyfit(literacy_north_africa, gdp_north_africa, 1)
line_literacy_gdp_north_africa= np.poly1d(coeffs_literacy_gdp_north_africa)
plt.subplot(2,1,2)
plt.scatter(literacy_north_africa,gdp_north_africa)
plt.plot(literacy_north_africa,line_literacy_gdp_north_africa(literacy_north_africa),color='red')

corr_coef_literacy_gdp_north_africa = np.corrcoef(literacy_north_africa, gdp_north_africa)[0, 1]

plt.xlabel('Literacy')
plt.ylabel('GDP')
plt.text(0.1, 0.6, f"y = {line_literacy_gdp_north_africa} \nCorrelation coefficient = {corr_coef_literacy_gdp_north_africa:.2f}", transform=plt.gca().transAxes)
plt.subplots_adjust(hspace=0.5)
plt.show(block=False)

plt.figure()


#Climate and GDP
coeffs_climate_gdp_north_africa = np.polyfit(climate_north_africa, gdp_north_africa, 1)
line_climate_gdp_north_africa = np.poly1d(coeffs_climate_gdp_north_africa)

corr_coef_climate_gdp_north_africa = np.corrcoef(climate_north_africa, gdp_north_africa)[0, 1]

plt.subplot(2,1,1)
plt.scatter(climate_north_africa, gdp_north_africa)
plt.plot(climate_north_africa, line_climate_gdp_north_africa(climate_north_africa), color='red')

plt.xlabel('Climate')
plt.ylabel('GDP')
plt.title('North Africa')
plt.text(0.5, 0.2, f"y = {line_climate_gdp_north_africa} \nCorrelation Coefficient = {corr_coef_climate_gdp_north_africa:.2f}", transform=plt.gca().transAxes)
plt.subplots_adjust(hspace=0.5)
plt.show(block=False)



#Birthrate and GDP
coeffs_birthrate_gdp_north_africa = np.polyfit(birthrate_north_africa, gdp_north_africa, 1)
line_birthrate_gdp_north_africa = np.poly1d(coeffs_birthrate_gdp_north_africa)

corr_coef_birthrate_gdp_north_africa = np.corrcoef(birthrate_north_africa, gdp_north_africa)[0, 1]

plt.subplot(2,1,2)
plt.scatter(birthrate_north_africa, gdp_north_africa)
plt.plot(birthrate_north_africa, line_birthrate_gdp_north_africa(birthrate_north_africa), color='red')

plt.xlabel('Birthrate')
plt.ylabel('GDP')
plt.text(0.1, 0.6, f"y = {line_birthrate_gdp_north_africa} \nCorrelation Coefficient = {corr_coef_birthrate_gdp_north_africa:.2f}", transform=plt.gca().transAxes)
plt.show(block=False)

plt.figure()
#Deathrate and GDP
coeffs_deathrate_gdp_north_africa = np.polyfit(deathrate_north_africa, gdp_north_africa, 1)
line_deathrate_gdp_north_africa = np.poly1d(coeffs_deathrate_gdp_north_africa)

corr_coef_deathrate_gdp_north_africa = np.corrcoef(deathrate_north_africa, gdp_north_africa)[0, 1]

plt.subplot(2,1,1)
plt.scatter(deathrate_north_africa, gdp_north_africa)
plt.plot(deathrate_north_africa, line_deathrate_gdp_north_africa(deathrate_north_africa), color='red')

plt.xlabel('Deathrate')
plt.ylabel('GDP')
plt.title('North Africa')
plt.text(0.1, 0.6, f"y = {line_deathrate_gdp_north_africa} \nCorrelation Coefficient = {corr_coef_deathrate_gdp_north_africa:.2f}", transform=plt.gca().transAxes)
plt.show(block=False)

#-------------North America---------------#

north_america = pd.read_csv('north_america.csv')

#North America x
population_north_america  = north_america['population'].values
literacy_north_america  = north_america['literacy'].values
climate_north_america  =north_america['climate'].values
birthrate_north_america  = north_america['birthrate'].values
deathrate_north_america  = north_america['deathrate'].values

#North America y
gdp_north_america  = north_america ['gdp'].values

#Population & GDP
coeffs_population_gdp_north_america  = np.polyfit(population_north_america , gdp_north_america , 1)
line_population_gdp_north_america = np.poly1d(coeffs_population_gdp_north_america )

corr_coef_pop_gdp_north_america = np.corrcoef(population_north_america , gdp_north_america )[0, 1]

plt.figure()
plt.subplot(2, 1, 1)
plt.scatter(population_north_america , gdp_north_america )
plt.plot(population_north_america , line_population_gdp_north_america (population_north_america ), color='red')

plt.xlabel('Population')
plt.ylabel('GDP')
plt.title('North America')
plt.text(0.5, 0.3, f"y = {line_population_gdp_north_america } \nCorrelation Coefficient = {corr_coef_pop_gdp_north_america :.2f}", transform=plt.gca().transAxes)


#Literacy & GDP
coeffs_literacy_gdp_north_america  = np.polyfit(literacy_north_america , gdp_north_america , 1)
line_literacy_gdp_north_america = np.poly1d(coeffs_literacy_gdp_north_america )
plt.subplot(2,1,2)
plt.scatter(literacy_north_america ,gdp_north_america )
plt.plot(literacy_north_america ,line_literacy_gdp_north_america (literacy_north_america ),color='red')

corr_coef_literacy_gdp_north_america  = np.corrcoef(literacy_north_america , gdp_north_america )[0, 1]

plt.xlabel('Literacy')
plt.ylabel('GDP')
plt.text(0.1, 0.6, f"y = {line_literacy_gdp_north_america } \nCorrelation coefficient = {corr_coef_literacy_gdp_north_america :.2f}", transform=plt.gca().transAxes)
plt.subplots_adjust(hspace=0.5)
plt.show(block=False)

plt.figure()


#Climate and GDP
coeffs_climate_gdp_north_america = np.polyfit(climate_north_america , gdp_north_america , 1)
line_climate_gdp_north_america  = np.poly1d(coeffs_climate_gdp_north_america )

corr_coef_climate_gdp_north_america  = np.corrcoef(climate_north_america , gdp_north_america )[0, 1]

plt.subplot(2,1,1)
plt.scatter(climate_north_america , gdp_north_america )
plt.plot(climate_north_america , line_climate_gdp_north_america (climate_north_america ), color='red')

plt.xlabel('Climate')
plt.ylabel('GDP')
plt.title('North America')
plt.text(0.5, 0.3, f"y = {line_climate_gdp_north_america } \nCorrelation Coefficient = {corr_coef_climate_gdp_north_america :.2f}", transform=plt.gca().transAxes)
plt.subplots_adjust(hspace=0.5)
plt.show(block=False)



#Birthrate and GDP
coeffs_birthrate_gdp_north_america  = np.polyfit(birthrate_north_america , gdp_north_america , 1)
line_birthrate_gdp_north_america  = np.poly1d(coeffs_birthrate_gdp_north_america )

corr_coef_birthrate_gdp_north_america  = np.corrcoef(birthrate_north_america , gdp_north_america )[0, 1]

plt.subplot(2,1,2)
plt.scatter(birthrate_north_america , gdp_north_america )
plt.plot(birthrate_north_america , line_birthrate_gdp_north_america (birthrate_north_america ), color='red')

plt.xlabel('Birthrate')
plt.ylabel('GDP')
plt.text(0.2, 0.3, f"y = {line_birthrate_gdp_north_america } \nCorrelation Coefficient = {corr_coef_birthrate_gdp_north_america :.2f}", transform=plt.gca().transAxes)
plt.show(block=False)

plt.figure()
#Deathrate and GDP
coeffs_deathrate_gdp_north_america  = np.polyfit(deathrate_north_america , gdp_north_america , 1)
line_deathrate_gdp_north_america  = np.poly1d(coeffs_deathrate_gdp_north_america)

corr_coef_deathrate_gdp_north_america = np.corrcoef(deathrate_north_america, gdp_north_america)[0, 1]

plt.subplot(2,1,1)
plt.scatter(deathrate_north_america, gdp_north_america)
plt.plot(deathrate_north_america, line_deathrate_gdp_north_america(deathrate_north_america), color='red')

plt.xlabel('Deathrate')
plt.ylabel('GDP')
plt.title('North America')
plt.text(0.1, 0.2, f"y = {line_deathrate_gdp_north_america} \nCorrelation Coefficient = {corr_coef_deathrate_gdp_north_america:.2f}", transform=plt.gca().transAxes)
plt.show(block=False)


#-------------Oceania---------------#

oceania = pd.read_csv('oceania.csv')

#Oceania x
population_oceania  = oceania['population'].values
literacy_oceania  = oceania['literacy'].values
climate_oceania  =oceania['climate'].values
birthrate_oceania  = oceania['birthrate'].values
deathrate_oceania  = oceania['deathrate'].values

#Oceania y
gdp_oceania  = oceania['gdp'].values

#Population & GDP
coeffs_population_gdp_oceania = np.polyfit(population_oceania , gdp_oceania, 1)
line_population_gdp_oceania = np.poly1d(coeffs_population_gdp_oceania)

corr_coef_pop_gdp_oceania= np.corrcoef(population_oceania , gdp_oceania )[0, 1]

plt.figure()
plt.subplot(2, 1, 1)
plt.scatter(population_oceania , gdp_oceania )
plt.plot(population_oceania , line_population_gdp_oceania (population_oceania ), color='red')

plt.xlabel('Population')
plt.ylabel('GDP')
plt.title('Oceania')
plt.text(0.5, 0.2, f"y = {line_population_gdp_oceania} \nCorrelation Coefficient = {corr_coef_pop_gdp_oceania:.2f}", transform=plt.gca().transAxes)


#Literacy & GDP
coeffs_literacy_gdp_oceania = np.polyfit(literacy_oceania, gdp_oceania, 1)
line_literacy_gdp_oceania= np.poly1d(coeffs_literacy_gdp_oceania)
plt.subplot(2,1,2)
plt.scatter(literacy_oceania,gdp_oceania)
plt.plot(literacy_oceania,line_literacy_gdp_oceania (literacy_oceania),color='red')

corr_coef_literacy_gdp_oceania = np.corrcoef(literacy_oceania, gdp_oceania)[0, 1]

plt.xlabel('Literacy')
plt.ylabel('GDP')
plt.text(0.1, 0.6, f"y = {line_literacy_gdp_oceania} \nCorrelation coefficient = {corr_coef_literacy_gdp_oceania:.2f}", transform=plt.gca().transAxes)
plt.subplots_adjust(hspace=0.5)
plt.show(block=False)

plt.figure()


#Climate and GDP
coeffs_climate_gdp_oceania = np.polyfit(climate_oceania , gdp_oceania , 1)
line_climate_gdp_oceania  = np.poly1d(coeffs_climate_gdp_oceania)

corr_coef_climate_gdp_oceania = np.corrcoef(climate_oceania, gdp_oceania)[0, 1]

plt.subplot(2,1,1)
plt.scatter(climate_oceania, gdp_oceania)
plt.plot(climate_oceania, line_climate_gdp_oceania (climate_oceania ), color='red')

plt.xlabel('Climate')
plt.ylabel('GDP')
plt.title('Oceania')
plt.text(0.5, 0.6, f"y = {line_climate_gdp_oceania} \nCorrelation Coefficient = {corr_coef_climate_gdp_oceania:.2f}", transform=plt.gca().transAxes)
plt.subplots_adjust(hspace=0.5)
plt.show(block=False)



#Birthrate and GDP
coeffs_birthrate_gdp_oceania= np.polyfit(birthrate_oceania, gdp_oceania, 1)
line_birthrate_gdp_oceania  = np.poly1d(coeffs_birthrate_gdp_oceania )

corr_coef_birthrate_gdp_oceania  = np.corrcoef(birthrate_oceania , gdp_oceania)[0, 1]

plt.subplot(2,1,2)
plt.scatter(birthrate_oceania, gdp_oceania)
plt.plot(birthrate_oceania, line_birthrate_gdp_oceania(birthrate_oceania), color='red')

plt.xlabel('Birthrate')
plt.ylabel('GDP')
plt.text(0.5, 0.6, f"y = {line_birthrate_gdp_oceania} \nCorrelation Coefficient = {corr_coef_birthrate_gdp_oceania:.2f}", transform=plt.gca().transAxes)
plt.show(block=False)

plt.figure()
#Deathrate and GDP
coeffs_deathrate_gdp_oceania  = np.polyfit(deathrate_oceania, gdp_oceania, 1)
line_deathrate_gdp_oceania = np.poly1d(coeffs_deathrate_gdp_oceania)

corr_coef_deathrate_gdp_oceania = np.corrcoef(deathrate_oceania, gdp_oceania)[0, 1]

plt.subplot(2,1,1)
plt.scatter(deathrate_oceania, gdp_oceania)
plt.plot(deathrate_oceania, line_deathrate_gdp_oceania(deathrate_oceania), color='red')

plt.xlabel('Deathrate')
plt.ylabel('GDP')
plt.title('Oceania')
plt.text(0.1, 0.6, f"y = {line_deathrate_gdp_oceania} \nCorrelation Coefficient = {corr_coef_deathrate_gdp_oceania:.2f}", transform=plt.gca().transAxes)
plt.show(block=False)

#-------------Subsahara---------------#

subsahara = pd.read_csv('subsahara.csv')

#Oceania x
population_subsahara  = subsahara['population'].values
literacy_subsahara  = subsahara['literacy'].values
climate_subsahara  =subsahara['climate'].values
birthrate_subsahara  = subsahara['birthrate'].values
deathrate_subsahara  = subsahara['deathrate'].values

#Oceania y
gdp_subsahara  = subsahara['gdp'].values

#Population & GDP
coeffs_population_gdp_subsahara = np.polyfit(population_subsahara , gdp_subsahara, 1)
line_population_gdp_subsahara = np.poly1d(coeffs_population_gdp_subsahara)

corr_coef_pop_gdp_subsahara= np.corrcoef(population_subsahara , gdp_subsahara)[0, 1]

plt.figure()
plt.subplot(2, 1, 1)
plt.scatter(population_subsahara, gdp_subsahara)
plt.plot(population_subsahara, line_population_gdp_subsahara(population_subsahara), color='red')

plt.xlabel('Population')
plt.ylabel('GDP')
plt.title('Subsahara')
plt.text(0.5, 0.5, f"y = {line_population_gdp_subsahara} \nCorrelation Coefficient = {corr_coef_pop_gdp_subsahara:.2f}", transform=plt.gca().transAxes)


#Literacy & GDP
coeffs_literacy_gdp_subsahara = np.polyfit(literacy_subsahara, gdp_subsahara, 1)
line_literacy_gdp_subsahara= np.poly1d(coeffs_literacy_gdp_subsahara)
plt.subplot(2,1,2)
plt.scatter(literacy_subsahara,gdp_subsahara)
plt.plot(literacy_subsahara,line_literacy_gdp_subsahara(literacy_subsahara),color='red')

corr_coef_literacy_gdp_subsahara= np.corrcoef(literacy_subsahara, gdp_subsahara)[0, 1]

plt.xlabel('Literacy')
plt.ylabel('GDP')
plt.text(0.1, 0.6, f"y = {line_literacy_gdp_subsahara} \nCorrelation coefficient = {corr_coef_literacy_gdp_subsahara:.2f}", transform=plt.gca().transAxes)
plt.subplots_adjust(hspace=0.5)
plt.show(block=False)

plt.figure()


#Climate and GDP
coeffs_climate_gdp_subsahara = np.polyfit(climate_subsahara, gdp_subsahara , 1)
line_climate_gdp_subsahara  = np.poly1d(coeffs_climate_gdp_subsahara)

corr_coef_climate_gdp_subsahara = np.corrcoef(climate_subsahara, gdp_subsahara)[0, 1]

plt.subplot(2,1,1)
plt.scatter(climate_subsahara, gdp_subsahara)
plt.plot(climate_subsahara, line_climate_gdp_subsahara(climate_subsahara), color='red')

plt.xlabel('Climate')
plt.ylabel('GDP')
plt.title('Subsahara')
plt.text(0.5, 0.6, f"y = {line_climate_gdp_subsahara} \nCorrelation Coefficient = {corr_coef_climate_gdp_subsahara:.2f}", transform=plt.gca().transAxes)
plt.subplots_adjust(hspace=0.5)
plt.show(block=False)



#Birthrate and GDP
coeffs_birthrate_gdp_subsahara= np.polyfit(birthrate_subsahara, gdp_subsahara, 1)
line_birthrate_gdp_subsahara= np.poly1d(coeffs_birthrate_gdp_subsahara)

corr_coef_birthrate_gdp_subsahara  = np.corrcoef(birthrate_subsahara, gdp_subsahara)[0, 1]

plt.subplot(2,1,2)
plt.scatter(birthrate_subsahara, gdp_subsahara)
plt.plot(birthrate_subsahara, line_birthrate_gdp_subsahara(birthrate_subsahara), color='red')

plt.xlabel('Birthrate')
plt.ylabel('GDP')
plt.text(0.5, 0.6, f"y = {line_birthrate_gdp_subsahara} \nCorrelation Coefficient = {corr_coef_birthrate_gdp_subsahara:.2f}", transform=plt.gca().transAxes)
plt.show(block=False)

plt.figure()
#Deathrate and GDP
coeffs_deathrate_gdp_subsahara  = np.polyfit(deathrate_subsahara, gdp_subsahara, 1)
line_deathrate_gdp_subsahara= np.poly1d(coeffs_deathrate_gdp_subsahara)

corr_coef_deathrate_gdp_subsahara= np.corrcoef(deathrate_subsahara, gdp_subsahara)[0, 1]

plt.subplot(2,1,1)
plt.scatter(deathrate_subsahara, gdp_subsahara)
plt.plot(deathrate_subsahara, line_deathrate_gdp_subsahara(deathrate_subsahara), color='red')

plt.xlabel('Deathrate')
plt.ylabel('GDP')
plt.title('Subsahara')
plt.text(0.5, 0.7, f"y = {line_deathrate_gdp_subsahara} \nCorrelation Coefficient = {corr_coef_deathrate_gdp_subsahara:.2f}", transform=plt.gca().transAxes)
plt.show(block=False)