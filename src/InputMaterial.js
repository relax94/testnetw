import React, {Component} from 'react'


export class InputMaterial extends Component{

    render(){
        return(
            <div className="group">
                <input type="text" required/>
                <span className="highlight"></span>
                <span className="bar"></span>
                <label>{this.props.name}</label>
            </div>
        )
    }

}